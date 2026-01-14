from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from mesh_to_sdf.depth_point_cloud import DepthPointCloud

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
os.environ["PYOPENGL_PLATFORM"] = "win32"

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional
import matplotlib.pyplot as plt

def save_point_cloud_as_ply(vertices, filename, colors=None):
    """
    Save a point cloud (with optional RGB colors) as a PLY file using Open3D.
    Args:
        vertices (numpy.ndarray): Nx3 array of 3D points.
        filename (str): Name of the output PLY file (without extension).
        colors (numpy.ndarray, optional): Nx3 array of RGB colors (values in range [0, 255]).
                                            If None, saves the point cloud without colors.        
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Add colors if provided
    if colors is not None:
        if colors.shape[0] != vertices.shape[0]:
            raise ValueError("The number of color entries must match the number of vertices.")
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB to [0, 1]    

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to '{filename}' using Open3D.")


def project_point(K, p_cam):
    """
    K: (3,3)
    p_cam: (3,) in camera coordinates (X,Y,Z)
    returns: (u,v), valid (bool)
    """
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


def project_uvz(K, P_cam):
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    u = K[0,0] * (X / (Z + 1e-8)) + K[0,2]
    v = K[1,1] * (Y / (Z + 1e-8)) + K[1,2]
    return u, v, Z

def depth_median_patch(depth, u, v, r=1, z_min=0.1, z_max=5.0):
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= W or vi < 0 or vi >= H:
        return None
    u0, u1 = max(0, ui-r), min(W, ui+r+1)
    v0, v1 = max(0, vi-r), min(H, vi+r+1)
    patch = depth[v0:v1, u0:u1].astype(np.float32)
    valid = (patch > z_min) & (patch < z_max)
    if not np.any(valid):
        return None
    return float(np.median(patch[valid]))


def visible_vertices_from_render_depth(vertices, x, K, depth_view,
                                       z_eps=0.008,
                                       invalid_is_zero=True):
    """
    vertices: (N,3) MANO vertices in model space (AFTER your right-hand x flip, if you do that)
    x: (3,) translation in camera frame (same frame as depth_view)
    K: (3,3) intrinsics used for rendering/projection (must match render_res)
    depth_view: (H,W) z-buffer depth from renderer for THIS pose
    z_eps: depth tolerance in meters
    invalid_is_zero: if True, treat depth==0 as background/invalid

    returns:
      vis: (N,) boolean array
      uv:  (N,2) projected pixel coords (float)
    """
    H, W = depth_view.shape[:2]

    # vertices to camera frame under hypothesis x
    V = vertices + x.reshape(1, 3)

    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]
    valid_z = Z > 1e-6

    # project
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
    # projection 1
    V1 = vertices + translation
    x1 = K1 @ V1.T
    x1[0, :] /= x1[2, :]
    x1[1, :] /= x1[2, :]

    # projection 2
    V2 = vertices + x
    x2 = K2 @ V2.T
    x2[0, :] /= x2[2, :]
    x2[1, :] /= x2[2, :]

    # 3D distances
    distances, indices = kd_tree.query(V2)
    distances = distances.astype(np.float32).reshape(-1)
    error_3d = 100 * np.mean(distances)

    # error
    error_2d =  np.square(x1[:2] - x2[:2]).mean()
    print('error 2d', error_2d, 'error_3d', error_3d)
    return error_2d + error_3d


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    intrinsic_matrix = np.array([[283.075, 0.0, 319.661],
        [0.0, 283.035, 199.896],
        [0.0, 0.0, 1.0]])
    print(intrinsic_matrix)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        print(img_path)

        if 'left' not in str(img_path):
            continue

        img_cv2 = cv2.imread(str(img_path))
        im_width = img_cv2.shape[1]
        im_height = img_cv2.shape[0]

        # load depth
        depth_path = str(img_path).replace('left', 'depth')
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= 1000.0
        print(np.min(depth), np.max(depth))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            print(out.keys())                

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))                     

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            print('len all verts', len(all_verts))
            cam_view, _ = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

            # Optimize each hand one by one
            optimized_cam_t = []
            for hand_id, (verts_i, cam_t_i, right_i) in enumerate(zip(all_verts, all_cam_t, all_right)):
                print(f"\n[opt] hand {hand_id}/{len(all_verts)-1}, right={right_i}")

                # Render ONLY this hand to get a per-hand mask
                cam_view_i, depth_view_i = renderer.render_rgba_multiple(
                    [verts_i], cam_t=[cam_t_i], render_res=img_size[n], is_right=[right_i], **misc_args
                )

                # plt.figure()
                # plt.imshow(depth_view_i)  # BGRA->RGBA for plt if needed; adjust if colors look off
                # plt.title("depth_view_i")
                # plt.axis("off")
                # plt.show()

                # check wrist keypoint
                # ---- inside your per-hand loop (hand_id / n) ----
                # right_i is 0/1 (numpy) in your loop; convert to int
                right_i_int = int(right_i)
                sign = (2 * right_i_int - 1)

                # wrist 3D in MANO space from HaMeR keypoints
                kpts3d = out["pred_keypoints_3d"][hand_id].detach().cpu().numpy()   # (K,3)
                kpts3d[:, 0] *= sign                                          # match your vertex flip
                wrist_mano = kpts3d[0]                                        # wrist keypoint (usually index 0)

                # move to camera frame
                wrist_cam = wrist_mano + cam_t_i                              # (3,)              

                # project to image pixels
                K = np.array([[12500, 0, im_width / 2],
                            [0, 12500, im_height / 2],
                            [0, 0, 1]], dtype=np.float32)                
                (u, v), ok = project_point(K, wrist_cam)

                # draw on cam_view_i (alpha mask / RGBA from renderer)
                cam_vis = (cam_view_i.copy() * 255).astype(np.uint8)          # RGBA uint8, shape (H,W,4)
                H, W = cam_vis.shape[:2]

                if ok and (0 <= u < W) and (0 <= v < H):
                    cv2.circle(cam_vis, (int(round(u)), int(round(v))), 6, (0, 0, 255, 255), -1)  # red dot
                    cv2.putText(cam_vis, "wrist", (int(round(u))+8, int(round(v))-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 255), 2)

                # get the hand points (mask from alpha)
                mask = cam_view_i[:, :, 3] > 0

                # wrist_cam is (X,Y,Z) in meters
                z = float(wrist_cam[2])

                # Example sensor size guess:
                # VRTrix puck/strap region: choose r_m ~ 0.03 to 0.05 m (3–5 cm)
                r_px = radius_px_from_metric(K[0,0], z_m=z, r_m=0.04)
                mask = mask.astype(np.uint8)
                cv2.circle(mask, (int(u), int(v)), r_px, 0, thickness=-1)                

                # erode mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel)

                # (keep your original inversion behavior)
                mask = 1 - mask

                # convert depth to point cloud for this hand
                depth_pc = DepthPointCloud(
                    depth, intrinsic_matrix,
                    camera_pose=np.eye(4),
                    target_mask=mask,
                    threshold=10.0,
                    use_kmeans=False
                )

                if depth_pc.points is None or len(depth_pc.points) < 50:
                    print(f"[opt] hand {hand_id}: too few depth points ({0 if depth_pc.points is None else len(depth_pc.points)}), skip.")
                    optimized_cam_t.append(cam_t_i)
                    continue

                print(f"[opt] hand {hand_id}: depth points = {depth_pc.points.shape}")

                # only select visible vertices
                vis, _ = visible_vertices_from_render_depth(verts_i, cam_t_i, K, depth_view_i, z_eps=0.008)
                print(f'for {verts_i.shape[0]} hand vertices, {vis.sum()} visible')
                if vis.sum() < 50:
                    vis[:] = True
                vertices = verts_i[vis]

                # solve new translation (one hand)
                x0 = np.mean(depth_pc.points, axis=0)
                res = minimize(
                    obj_funcion, x0, method='nelder-mead',
                    args=(vertices, cam_t_i, K, intrinsic_matrix, depth_pc.kd_tree),
                    options={'xatol': 1e-8, 'disp': True}
                )
                translation_new = res.x
                optimized_cam_t.append(translation_new)
                print(f"[opt] hand {hand_id}: translation_new = {translation_new}")

                # --- Debug visualization per hand (optional but useful) ---
                # --- Debug visualization per hand ---
                fig = plt.figure(figsize=(14, 8))

                # 1) Input image
                ax = fig.add_subplot(2, 3, 1)
                ax.imshow(input_img)
                ax.set_title(f'input (hand {hand_id})')
                ax.axis('off')

                # 2) Mask visualization (IMPORTANT)
                ax = fig.add_subplot(2, 3, 2)
                ax.imshow(mask, cmap='gray')
                ax.set_title(f'depth mask (hand {hand_id})')
                ax.axis('off')

                # 3) HaMeR projection
                ax = fig.add_subplot(2, 3, 3)
                ax.imshow(input_img)
                vertices_hamer = vertices + cam_t_i
                x2d = K @ vertices_hamer.T
                x2d[0, :] /= x2d[2, :]
                x2d[1, :] /= x2d[2, :]
                ax.plot(x2d[0, :], x2d[1, :], linewidth=0.5)
                ax.set_title(f'proj using HaMeR camera')
                ax.axis('off')

                # 4) Optimized projection
                ax = fig.add_subplot(2, 3, 4)
                ax.imshow(input_img)
                vertices_opt = vertices + translation_new
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
                ax.imshow(cam_vis[:, :, [2,1,0,3]])  # BGRA->RGBA for plt if needed; adjust if colors look off
                ax.set_title("cam_view_i + wrist projection")
                ax.axis("off")

                plt.tight_layout()
                plt.show()

                # save ply per hand
                depth_pc_full = DepthPointCloud(depth, intrinsic_matrix, camera_pose=np.eye(4), target_mask=None, threshold=10.0)
                colors = np.zeros(depth_pc_full.points.shape)
                colors[:, 1] = 255
                save_point_cloud_as_ply(
                    depth_pc_full.points,
                    filename=os.path.join(args.out_folder, f'depth_hand{hand_id}.ply'),
                    colors=colors
                )

                colors = np.zeros(vertices_opt.shape)
                colors[:, 0] = 255
                save_point_cloud_as_ply(
                    vertices_opt,
                    filename=os.path.join(args.out_folder, f'hand_hand{hand_id}.ply'),
                    colors=colors
                )

if __name__ == '__main__':
    main()