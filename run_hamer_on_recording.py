#!/usr/bin/env python3
"""
run_hamer_on_recording_with_depth_opt.py

Run HaMeR hand pose estimation on a recording folder with:
  - rectified_left_frames/left_000000_t1.719302.png
  - depth_frames/depth_000000_t1.719302.png
  - calibration.json (under data_dir)

Pairs RGB/depth by matching the suffix after the prefix:
  left_<SUFFIX>.png  <->  depth_<SUFFIX>.png

Adds per-hand translation optimization using depth (your code).

Also saves the hand bounding boxes (xyxy in full RGB image coords) for each detected hand.

Outputs (under out_folder):
  - results.jsonl   (one line per frame; includes per-hand bbox_xyxy + cam_t_full + translation_opt)
  - renders/        (optional overlay renders)
  - meshes/         (optional .obj meshes)
"""

from pathlib import Path
import argparse
import os
import json
from typing import Tuple, List

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from mesh_to_sdf.depth_point_cloud import DepthPointCloud

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel


# -----------------------------
# Geometry helpers (from your code)
# -----------------------------
def project_point(K, p_cam):
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Z <= 1e-6:
        return (np.nan, np.nan), False
    u = K[0, 0] * (X / Z) + K[0, 2]
    v = K[1, 1] * (Y / Z) + K[1, 2]
    return (u, v), True


def radius_px_from_metric(fx, z_m, r_m, r_min=8, r_max=80):
    if z_m <= 1e-6:
        return r_min
    r_px = fx * (r_m / z_m)
    return int(np.clip(r_px, r_min, r_max))


def visible_vertices_from_render_depth(vertices, x, K, depth_view,
                                       z_eps=0.008,
                                       invalid_is_zero=True):
    H, W = depth_view.shape[:2]

    V = vertices + x.reshape(1, 3)
    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]
    valid_z = Z > 1e-6

    u = K[0, 0] * (X / (Z + 1e-8)) + K[0, 2]
    v = K[1, 1] * (Y / (Z + 1e-8)) + K[1, 2]
    uv = np.stack([u, v], axis=1)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inb = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    vis = np.zeros(vertices.shape[0], dtype=bool)

    idx = np.where(inb)[0]
    zbuf = depth_view[vi[idx], ui[idx]].astype(np.float32)

    if invalid_is_zero:
        ok_depth = zbuf > 0.0
    else:
        ok_depth = np.isfinite(zbuf)

    ok = ok_depth & (np.abs(Z[idx].astype(np.float32) - zbuf) <= float(z_eps))
    vis[idx[ok]] = True
    return vis, uv


def obj_funcion(x, vertices, translation, K1, K2, kd_tree):
    V1 = vertices + translation
    x1 = K1 @ V1.T
    x1[0, :] /= x1[2, :]
    x1[1, :] /= x1[2, :]

    V2 = vertices + x
    x2 = K2 @ V2.T
    x2[0, :] /= x2[2, :]
    x2[1, :] /= x2[2, :]

    distances, _ = kd_tree.query(V2)
    distances = distances.astype(np.float32).reshape(-1)
    error_3d = 100 * np.mean(distances)

    error_2d = np.square(x1[:2] - x2[:2]).mean()
    return float(error_2d + error_3d)


# -----------------------------
# Pairing + IO
# -----------------------------
def build_pairs_prefix_suffix(
    data_dir: Path,
    rgb_subdir: str = "rectified_left_frames",
    depth_subdir: str = "depth_frames",
    rgb_prefix: str = "left_",
    depth_prefix: str = "depth_",
    ext: str = ".png",
) -> List[Tuple[Path, Path]]:
    rgb_dir = data_dir / rgb_subdir
    depth_dir = data_dir / depth_subdir
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth folder not found: {depth_dir}")

    rgb_paths = sorted(rgb_dir.glob(f"{rgb_prefix}*{ext}"))
    depth_paths = sorted(depth_dir.glob(f"{depth_prefix}*{ext}"))

    depth_map = {}
    for dp in depth_paths:
        name = dp.name
        if not name.startswith(depth_prefix):
            continue
        suffix = name[len(depth_prefix):]
        depth_map[suffix] = dp

    pairs = []
    missing = 0
    for rp in rgb_paths:
        name = rp.name
        if not name.startswith(rgb_prefix):
            continue
        suffix = name[len(rgb_prefix):]
        dp = depth_map.get(suffix, None)
        if dp is None:
            missing += 1
            continue
        pairs.append((rp, dp))

    print(f"[pairs] rgb={len(rgb_paths)} depth={len(depth_paths)} paired={len(pairs)} missing_depth_for_rgb={missing}")
    return pairs


def load_depth_meters(depth_path: Path) -> np.ndarray:
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    d = d.astype(np.float32)
    if np.nanmax(d) > 20.0:
        d *= 0.001
    return d


def save_jsonl_line(fp, obj: dict):
    fp.write(json.dumps(obj) + "\n")
    fp.flush()


# -----------------------------
# Calibration
# -----------------------------
def load_K_from_calibration_json(calib_path: Path, target_wh=None) -> np.ndarray:
    if not calib_path.exists():
        raise FileNotFoundError(f"calibration.json not found: {calib_path}")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    cams = calib.get("cameras", None)
    if not cams:
        raise ValueError(f"No 'cameras' array in {calib_path}")

    cam_sel = None
    if target_wh is not None:
        tw, th = int(target_wh[0]), int(target_wh[1])
        for c in cams:
            if int(c.get("imageWidth", -1)) == tw and int(c.get("imageHeight", -1)) == th:
                cam_sel = c
                break
    if cam_sel is None:
        cam_sel = cams[0]

    fx = float(cam_sel["focalLengthX"])
    fy = float(cam_sel["focalLengthY"])
    cx = float(cam_sel["principalPointX"])
    cy = float(cam_sel["principalPointY"])

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    print(f"[calib] using camera entry: {cam_sel.get('imageWidth')}x{cam_sel.get('imageHeight')}, "
          f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return K


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="HaMeR on recording folder + depth translation opt + save hand bboxes")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--rgb_dirname", type=str, default="rectified_left_frames")
    parser.add_argument("--depth_dirname", type=str, default="depth_frames")
    parser.add_argument("--rgb_prefix", type=str, default="left_")
    parser.add_argument("--depth_prefix", type=str, default="depth_")
    parser.add_argument("--ext", type=str, default=".png")

    parser.add_argument("--out_folder", type=str, default="out_hamer_recording")
    parser.add_argument("--save_renders", action="store_true", default=False)
    parser.add_argument("--save_full_frame", action="store_true", default=True)
    parser.add_argument("--save_mesh", action="store_true", default=False)
    parser.add_argument("--side_view", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--body_detector", type=str, default="vitdet", choices=["vitdet", "regnety"])

    # depth optimization knobs
    parser.add_argument("--do_opt", action="store_true", default=True)
    parser.add_argument("--opt_threshold", type=float, default=10.0)
    parser.add_argument("--opt_z_eps", type=float, default=0.008)
    parser.add_argument("--opt_min_points", type=int, default=50)
    parser.add_argument("--opt_r_m", type=float, default=0.04)
    parser.add_argument("--opt_erode", type=int, default=5)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "renders"
    mesh_dir = out_dir / "meshes"
    render_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mesh:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs_prefix_suffix(
        data_dir / "export_pairs",
        rgb_subdir=args.rgb_dirname,
        depth_subdir=args.depth_dirname,
        rgb_prefix=args.rgb_prefix,
        depth_prefix=args.depth_prefix,
        ext=args.ext,
    )
    if len(pairs) == 0:
        print("[error] no RGB-depth pairs found.")
        return

    calib_path = data_dir / "calibration.json"
    first_depth = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
    if first_depth is None:
        raise RuntimeError(f"Could not read first depth frame: {pairs[0][1]}")
    dh, dw = first_depth.shape[:2]
    intrinsic_matrix = load_K_from_calibration_json(calib_path, target_wh=(dw, dh))

    # HaMeR
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()

    # Detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
            trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    jsonl_path = out_dir / "results_hamer.jsonl"
    f_jsonl = open(jsonl_path, "w", encoding="utf-8")

    for frame_idx, (rgb_path, depth_path) in enumerate(pairs):

        frame_id = rgb_path.name
        if frame_id.startswith(args.rgb_prefix):
            frame_id = frame_id[len(args.rgb_prefix):]
        frame_id = Path(frame_id).stem

        print(f"\n[frame {frame_idx}/{len(pairs)-1}] id={frame_id}")
        img_bgr = cv2.imread(str(rgb_path))
        if img_bgr is None:
            print(f"[warn] could not read rgb: {rgb_path}, skip.")
            continue

        depth = load_depth_meters(depth_path)
        im_h, im_w = img_bgr.shape[:2]

        det_out = detector(img_bgr)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if pred_bboxes.shape[0] == 0:
            save_jsonl_line(f_jsonl, {
                "frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "image_size": [int(im_w), int(im_h)],
                "depth_size": [int(depth.shape[1]), int(depth.shape[0])],
                "depth_K": intrinsic_matrix.tolist(),
                "num_people": 0,
                "num_hands": 0,
                "hands": []
            })
            continue

        img_rgb = img_bgr[:, :, ::-1].copy()

        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # Hand bboxes from keypoints in full-image coords
        hand_bboxes = []   # list of bbox_xyxy
        hand_is_right = [] # 0/1

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                bbox = [float(keyp[valid, 0].min()), float(keyp[valid, 1].min()),
                        float(keyp[valid, 0].max()), float(keyp[valid, 1].max())]
                hand_bboxes.append(bbox)
                hand_is_right.append(0)

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                bbox = [float(keyp[valid, 0].min()), float(keyp[valid, 1].min()),
                        float(keyp[valid, 0].max()), float(keyp[valid, 1].max())]
                hand_bboxes.append(bbox)
                hand_is_right.append(1)

        if len(hand_bboxes) == 0:
            save_jsonl_line(f_jsonl, {
                "frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "image_size": [int(im_w), int(im_h)],
                "depth_size": [int(depth.shape[1]), int(depth.shape[0])],
                "depth_K": intrinsic_matrix.tolist(),
                "num_people": int(pred_bboxes.shape[0]),
                "num_hands": 0,
                "hands": []
            })
            continue

        boxes = np.array(hand_bboxes.copy(), dtype=np.float32)
        right = np.array(hand_is_right, dtype=np.int64)

        dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Cache for each hand, aligned with the dataset order
        hands_cache = []
        all_verts, all_cam_t, all_right = [], [], []
        scaled_focal_length = None

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            # mano parameters
            betas = out["pred_mano_params"]["betas"]
            global_orient = out["pred_mano_params"]["global_orient"]  # (B, 3)
            hand_pose = out["pred_mano_params"]["hand_pose"]          # (B, 45)
            # Full MANO pose (axis-angle per joint)
            mano_pose = torch.cat([global_orient, hand_pose], dim=1)  # (B, 48)        

            bs = batch["img"].shape[0]
            for n in range(bs):
                hid = int(batch["personid"][n])  # NOTE: this is the hand index in ViTDetDataset, despite the name
                is_r = int(batch["right"][n].detach().cpu().item())

                verts = out["pred_vertices"][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n].copy()

                # x-flip
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]

                kpts3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                kpts3d[:, 0] *= (2 * is_r - 1)

                hands_cache.append({
                    "hand_index": hid,
                    "is_right": is_r,
                    "bbox_xyxy": hand_bboxes[hid],
                    "cam_t_full": cam_t,
                    "pred_vertices": verts,
                    "pred_keypoints_3d": kpts3d,
                    "pred_vertices_raw": out["pred_vertices"][n].detach().cpu().numpy(),
                    "pred_cam_t_crop": out["pred_cam_t"][n].detach().cpu().numpy(),
                    "img_patch": batch["img"][n],
                    "mano_pose": mano_pose[n].detach().cpu().numpy(),
                    "mano_shape": betas[n].detach().cpu().numpy(),      
                })

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_r)

                if args.save_mesh:
                    tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE, is_right=is_r)
                    tmesh.export(str(mesh_dir / f"{frame_id}_hand{hid}.obj"))

                if args.save_renders:
                    white_img = (torch.ones_like(batch["img"][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                        DEFAULT_STD[:, None, None] / 255
                    )
                    input_patch = batch["img"][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                        DEFAULT_MEAN[:, None, None] / 255
                    )
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    regression_img = renderer(
                        hands_cache[-1]["pred_vertices_raw"],
                        hands_cache[-1]["pred_cam_t_crop"],
                        batch["img"][n],
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                    )

                    if args.side_view:
                        side_img = renderer(
                            hands_cache[-1]["pred_vertices_raw"],
                            hands_cache[-1]["pred_cam_t_crop"],
                            white_img,
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            side_view=True
                        )
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                    cv2.imwrite(
                        str(render_dir / f"{frame_id}_hand{hid}.png"),
                        (255 * final_img[:, :, ::-1]).astype(np.uint8)
                    )

        # Optional full-frame render
        if args.save_full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=float(scaled_focal_length) if scaled_focal_length is not None else None,
            )
            render_res = (im_w, im_h)
            cam_view, _ = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=render_res, is_right=all_right, **misc_args
            )
            input_img = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            cv2.imwrite(str(render_dir / f"{frame_id}_all.jpg"), (255 * overlay[:, :, ::-1]).astype(np.uint8))

        # Depth optimization
        optimized_translations = [None] * len(hand_bboxes)
        if args.do_opt and len(hands_cache) > 0:
            K_render = np.array([[12500.0, 0.0, im_w / 2.0],
                                 [0.0, 12500.0, im_h / 2.0],
                                 [0.0, 0.0, 1.0]], dtype=np.float32)

            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=float(scaled_focal_length) if scaled_focal_length is not None else None,
            )

            for h in hands_cache:
                hid = int(h["hand_index"])
                verts_i = h["pred_vertices"]
                cam_t_i = h["cam_t_full"]
                right_i = int(h["is_right"])

                cam_view_i, depth_view_i = renderer.render_rgba_multiple(
                    [verts_i], cam_t=[cam_t_i], render_res=(im_w, im_h), is_right=[right_i], **misc_args
                )

                kpts3d = h["pred_keypoints_3d"]
                wrist_mano = kpts3d[0]
                wrist_cam = wrist_mano + cam_t_i
                (u, v), ok = project_point(K_render, wrist_cam)
                if not ok:
                    continue

                mask = (cam_view_i[:, :, 3] > 0).astype(np.uint8)

                z = float(wrist_cam[2])
                r_px = radius_px_from_metric(K_render[0, 0], z_m=z, r_m=float(args.opt_r_m))
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < im_w and 0 <= vi < im_h:
                    cv2.circle(mask, (ui, vi), r_px, 0, thickness=-1)

                if args.opt_erode and args.opt_erode > 0:
                    ksz = int(args.opt_erode)
                    if ksz % 2 == 0:
                        ksz += 1
                    kernel = np.ones((ksz, ksz), np.uint8)
                    mask = cv2.erode(mask, kernel)

                mask = 1 - mask

                depth_pc = DepthPointCloud(
                    depth, intrinsic_matrix,
                    camera_pose=np.eye(4),
                    target_mask=mask,
                    threshold=float(args.opt_threshold),
                    use_kmeans=False
                )

                if depth_pc.points is None or len(depth_pc.points) < int(args.opt_min_points):
                    continue

                vis, _ = visible_vertices_from_render_depth(
                    verts_i, cam_t_i, K_render, depth_view_i, z_eps=float(args.opt_z_eps)
                )
                if int(vis.sum()) < int(args.opt_min_points):
                    vis[:] = True
                vertices_vis = verts_i[vis]

                x0 = np.mean(depth_pc.points, axis=0)
                try:
                    res = minimize(
                        obj_funcion, x0, method="nelder-mead",
                        args=(vertices_vis, cam_t_i, K_render, intrinsic_matrix, depth_pc.kd_tree),
                        options={"xatol": 1e-8, "disp": False}
                    )
                    optimized_translations[hid] = res.x.astype(np.float32).tolist()
                except Exception as e:
                    print(f"[opt] hand {hid}: minimize failed: {e}")

                if args.debug:
                    # --- Debug visualization per hand (optional but useful) ---
                    # --- Debug visualization per hand ---
                    fig = plt.figure(figsize=(14, 8))
                    # 1) Input image
                    ax = fig.add_subplot(2, 3, 1)
                    ax.imshow(input_img)
                    ax.set_title(f'input (hand {hid})')
                    ax.axis('off')

                    # 2) Mask visualization (IMPORTANT)
                    ax = fig.add_subplot(2, 3, 2)
                    ax.imshow(mask, cmap='gray')
                    ax.set_title(f'depth mask (hand {hid})')
                    ax.axis('off')

                    # 3) HaMeR projection
                    ax = fig.add_subplot(2, 3, 3)
                    ax.imshow(input_img)
                    vertices_hamer = vertices_vis + cam_t_i
                    x2d = K_render @ vertices_hamer.T
                    x2d[0, :] /= x2d[2, :]
                    x2d[1, :] /= x2d[2, :]
                    ax.plot(x2d[0, :], x2d[1, :], linewidth=0.5)
                    ax.set_title(f'proj using HaMeR camera')
                    ax.axis('off')

                    # 4) Optimized projection
                    ax = fig.add_subplot(2, 3, 4)
                    ax.imshow(input_img)
                    vertices_opt = vertices_vis + optimized_translations[hid]
                    x2d = intrinsic_matrix @ vertices_opt.T
                    x2d[0, :] /= x2d[2, :]
                    x2d[1, :] /= x2d[2, :]
                    ax.plot(x2d[0, :], x2d[1, :], linewidth=0.5)
                    ax.set_title(f'proj using OAK intrinsics')
                    ax.axis('off')

                    # 5) 3D alignment
                    ax = fig.add_subplot(2, 3, 5, projection='3d')
                    ax.scatter(depth_pc.points[:, 0],
                            depth_pc.points[:, 1],
                            depth_pc.points[:, 2],
                            marker='o', s=1)
                    ax.scatter(vertices_opt[:, 0],
                            vertices_opt[:, 1],
                            vertices_opt[:, 2],
                            marker='o', s=1, color='r')
                    ax.set_title('3D depth (gray) vs hand (red)')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')

                    # wrist
                    ax = fig.add_subplot(2, 3, 6)
                    cam_vis = (cam_view_i.copy() * 255).astype(np.uint8)          # RGBA uint8, shape (H,W,4)
                    ax.imshow(cam_vis[:, :, [2,1,0,3]])  # BGRA->RGBA for plt if needed; adjust if colors look off
                    ax.set_title("cam_view_i + wrist projection")
                    ax.axis("off")

                    plt.tight_layout()
                    plt.show()                    

        # Save frame output
        hands_out = []
        # sort by hand_index for stable output order
        hands_cache_sorted = sorted(hands_cache, key=lambda d: int(d["hand_index"]))
        for h in hands_cache_sorted:
            hid = int(h["hand_index"])
            cam_t = h["cam_t_full"].astype(np.float32)
            hands_out.append({
                "hand_index": hid,
                "is_right": int(h["is_right"]),
                "bbox_xyxy": [float(x) for x in h["bbox_xyxy"]],
                "mano_pose": h["mano_pose"].astype(np.float32).tolist(),
                "mano_shape": h["mano_shape"].astype(np.float32).tolist(),
                "cam_t_full": cam_t.tolist(),
                "translation_opt": optimized_translations[hid],
                "pred_vertices": h["pred_vertices"].astype(np.float32).tolist(),
                "pred_keypoints_3d": h["pred_keypoints_3d"].astype(np.float32).tolist(),
            })

        save_jsonl_line(f_jsonl, {
            "frame_id": frame_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "image_size": [int(im_w), int(im_h)],
            "depth_size": [int(depth.shape[1]), int(depth.shape[0])],
            "depth_K": intrinsic_matrix.tolist(),
            "num_people": int(pred_bboxes.shape[0]),
            "num_hands": int(len(hands_out)),
            "hands": hands_out
        })

    f_jsonl.close()
    print(f"\n[done] wrote: {jsonl_path}")
    print(f"[done] renders: {render_dir}")
    if args.save_mesh:
        print(f"[done] meshes: {mesh_dir}")


if __name__ == "__main__":
    main()