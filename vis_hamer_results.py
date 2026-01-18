#!/usr/bin/env python3
"""
vis_hamer_results.py

Visualize HaMeR results and draw bounding boxes computed from the *projected MANO vertices*
(not from the detector / keypoint bbox).

- Reads results.jsonl (preferred) OR results.json
- Shows frames one-by-one with:
    * RGB image
    * projected-vertex bbox per hand (xyxy computed after projection + in-bounds filtering)
    * optional vertex scatter overlay

Example:
  python vis_hamer_results.py --results out_hamer_recording/results.jsonl --data_root /path/to/recording
  python vis_hamer_results.py --results out_hamer_recording/results.jsonl --use_opt --show_vertices
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation
from manotorch.manolayer import ManoLayer, MANOOutput

import pyvista as pv

def plot_two_point_sets(
    V_hamer: np.ndarray,
    V_recon: np.ndarray,
    *,
    point_size: float = 4.0,
    title: str = "HaMeR pred_vertices (red) vs reconstructed MANO (blue)",
):
    """
    V_hamer: (N,3) from h["pred_vertices"]
    V_recon: (N,3) from ManoLayer reconstruction
    """
    assert V_hamer.shape == V_recon.shape

    pl = pv.Plotter(title=title)
    pl.set_background("white")

    # HaMeR vertices
    pl.add_points(
        V_hamer,
        color="red",
        point_size=point_size,
        render_points_as_spheres=True,
        label="HaMeR pred_vertices",
    )

    # Reconstructed MANO vertices
    pl.add_points(
        V_recon,
        color="blue",
        point_size=point_size,
        render_points_as_spheres=True,
        label="Reconstructed MANO",
    )

    axes = pv.Axes(show_actor=True)
    pl.add_actor(axes.actor)        

    pl.add_legend()
    pl.add_camera_orientation_widget()
    pl.show()


def make_mano_layers(device: torch.device):
    mano_left = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="left",
        center_idx=0,
        mano_assets_root="assets/mano",
        flat_hand_mean=True,
    ).to(device)

    mano_right = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=0,
        mano_assets_root="assets/mano",
        flat_hand_mean=True,
    ).to(device)

    return mano_left, mano_right


def mano_rotmats16_to_axisang48(mano_pose_rotmats: np.ndarray) -> np.ndarray:
    """
    mano_pose_rotmats: (16,3,3) or (B,16,3,3)
    returns axisang: (48,) or (B,48)
    """
    Rm = np.asarray(mano_pose_rotmats, dtype=np.float32)
    if Rm.ndim == 3:
        Rm = Rm[None, ...]  # (1,16,3,3)
    assert Rm.shape[1:] == (16, 3, 3)

    B = Rm.shape[0]
    aa = np.zeros((B, 16, 3), dtype=np.float32)
    for b in range(B):
        aa[b] = Rotation.from_matrix(Rm[b]).as_rotvec().astype(np.float32)  # (16,3)

    aa48 = aa.reshape(B, 48)
    return aa48[0] if mano_pose_rotmats.ndim == 3 else aa48


def hamer_hand_to_mano_out(
    h: Dict[str, Any],
    *,
    mano_right: ManoLayer,
    device: torch.device,
    mirror_left_x: bool = True,
) -> Tuple[MANOOutput, Dict[str, Any]]:
    """
    Convert one HaMeR hand record `h` (from rec["hands"]) to manotorch MANOOutput.

    Assumptions (matches your current export):
      - h["mano_pose"] is (16,3,3) rotation matrices in MANO joint order (root + 15).
      - h["mano_shape"] is (10,) or (1,10) betas.
      - h["is_right"] exists: 1 for right, 0 for left.
      - HaMeR uses right-hand MANO model; left is handled by mirroring X if desired.

    Returns:
      mano_out: MANOOutput from manotorch
      info: dict with useful extras:
        - lr: "R" or "L"
        - did_mirror: bool
        - V_local: (778,3) vertices in MANO root frame (after optional mirroring)
        - V_wrist_canon: (778,3) vertices in wrist-canonical frame (after optional mirroring)
        - R_wrist, t_wrist (numpy)
    """
    mano_pose = h.get("mano_pose", None)
    mano_shape = h.get("mano_shape", None)
    if mano_pose is None or mano_shape is None:
        raise ValueError("HaMeR hand record missing mano_pose or mano_shape")

    mano_pose = np.asarray(mano_pose, dtype=np.float32)
    mano_shape = np.asarray(mano_shape, dtype=np.float32)

    lr = "R" if int(h.get("is_right", 0)) == 1 else "L"

    # Convert pose rotmats -> axis-angle 48
    pose_aa48 = mano_rotmats16_to_axisang48(mano_pose)  # (48,)
    pose_t = torch.from_numpy(pose_aa48).float().view(1, 48).to(device)

    # Shape (1,10)
    shape_t = torch.from_numpy(mano_shape).float()
    if shape_t.ndim == 1:
        shape_t = shape_t.view(1, -1)
    shape_t = shape_t[:, :10].to(device)

    # HaMeR: always decode with RIGHT-hand model
    mano_out: MANOOutput = mano_right(pose_t, shape_t)

    # Extract vertices and wrist FK
    V = mano_out.verts[0]  # (778,3) torch
    T_abs = mano_out.transforms_abs[0]  # (16,4,4) torch
    R_wrist_t = T_abs[0, :3, :3]
    t_wrist_t = T_abs[0, :3, 3]

    did_mirror = False

    # Optional: mirror left hand in X to match your convention
    if lr == "L" and mirror_left_x:
        did_mirror = True

        # Mirror vertices in MANO root frame
        V = V.clone()
        V[:, 0] *= -1.0

        # Mirror wrist transform consistently: x' = -x  ->  M = diag([-1,1,1])
        M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device))
        R_wrist_t = M @ R_wrist_t @ M  # conjugation
        t_wrist_t = (M @ t_wrist_t)

    # Wrist-canonicalize: express verts in wrist frame
    V_wrist_canon_t = (V - t_wrist_t[None, :]) @ R_wrist_t  # (778,3)

    info = {
        "lr": lr,
        "did_mirror": did_mirror,
        "V_local": V.detach().cpu().numpy().astype(np.float64),
        "V_wrist_canon": V_wrist_canon_t.detach().cpu().numpy().astype(np.float64),
        "R_wrist": R_wrist_t.detach().cpu().numpy().astype(np.float64),
        "t_wrist": t_wrist_t.detach().cpu().numpy().astype(np.float64),
    }
    return mano_out, info


def iter_results(path: Path):
    """
    Supports:
      - .jsonl: one JSON object per line (frame record)
      - .json : either a list of frame records, or a dict containing "frames"
    """
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif suf == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                yield rec
        elif isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
            for rec in data["frames"]:
                yield rec
        else:
            raise ValueError(f"Unrecognized JSON format in {path}")
    else:
        raise ValueError(f"Unsupported results file suffix: {path.suffix}")


def project_points(K: np.ndarray, P: np.ndarray):
    """
    K: (3,3)
    P: (N,3) in camera coordinates
    Returns u, v, valid_z
    """
    X = P[:, 0]
    Y = P[:, 1]
    Z = P[:, 2]
    valid = Z > 1e-6
    u = K[0, 0] * (X / (Z + 1e-8)) + K[0, 2]
    v = K[1, 1] * (Y / (Z + 1e-8)) + K[1, 2]
    return u, v, valid


def bbox_from_uv(u: np.ndarray, v: np.ndarray, W: int, H: int, margin_px: int = 2):
    """
    Compute xyxy bbox from projected points.
    Keeps only in-bounds points.
    Returns [x1,y1,x2,y2] or None.
    """
    if u.size == 0:
        return None
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(inb):
        return None
    uu = u[inb]
    vv = v[inb]
    x1 = float(np.clip(np.min(uu) - margin_px, 0, W - 1))
    y1 = float(np.clip(np.min(vv) - margin_px, 0, H - 1))
    x2 = float(np.clip(np.max(uu) + margin_px, 0, W - 1))
    y2 = float(np.clip(np.max(vv) + margin_px, 0, H - 1))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return [x1, y1, x2, y2]


def load_bgr(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True,
                    help="Path to results.jsonl (preferred) or results.json")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Optional: prefix to prepend to rgb_path if paths in results are relative")
    ap.add_argument("--show_vertices", action="store_true", default=True,
                    help="If set, overlay projected vertices (can be slow)")
    ap.add_argument("--max_vertices", type=int, default=1500,
                    help="Subsample vertices for speed (0 = no subsample)")
    ap.add_argument("--margin_px", type=int, default=2,
                    help="Extra margin added to projected bbox")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="If > 0, auto-advance after this many seconds; otherwise wait for window close")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")    
    # NOTE: we intentionally do NOT draw saved detection bboxes.
    args = ap.parse_args()

    results_path = Path(args.results)
    data_root = Path(args.data_root) if args.data_root else None

    # device for MANO
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("[mano] device:", device)    

    # ---- MANO layers (also used to get J_regressor for wrist-centering)
    mano_left, mano_right = make_mano_layers(device)    

    for i, rec in enumerate(iter_results(results_path)):
        rgb_path = Path(rec["rgb_path"])
        if not rgb_path.is_absolute() and data_root is not None:
            rgb_path = data_root / rgb_path

        img_bgr = load_bgr(rgb_path)
        img_rgb = img_bgr[:, :, ::-1]
        H, W = img_rgb.shape[:2]
        K = rec["depth_K"]
        K = np.asarray(K, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_rgb)
        ax.set_title(f"{i}: frame_id={rec.get('frame_id', '')}  hands={rec.get('num_hands', 0)}")
        ax.axis("off")

        hands = rec.get("hands", [])

        for h in hands:
            hid = h.get("hand_index", "?")
            lr = "R" if int(h.get("is_right", 0)) == 1 else "L"

            verts = h.get("pred_vertices", None)
            if verts is None:
                continue
            verts = np.asarray(verts, dtype=np.float32)
            print(verts.shape) 

            # For hamer, always use right model (your convention)
            mano_out, info = hamer_hand_to_mano_out(h, mano_right=mano_right, device=device, mirror_left_x=True)
            V_local = info["V_local"]                 # (778,3)
            V_canon = info["V_wrist_canon"]          # (778,3)
       

            V_hamer = np.asarray(h["pred_vertices"], dtype=np.float64)
            K3d = np.asarray(h["pred_keypoints_3d"], dtype=np.float32)    # (21,3) usually
            wrist = K3d[0]                                                # (3,)
            V_wrist = V_hamer - wrist[None, :]                 
            plot_two_point_sets(V_wrist, V_canon, point_size=3.0)

            # Subsample for speed (bbox computed on subsample is still usually fine;
            # set --max_vertices 0 if you want exact bbox)
            if args.max_vertices and args.max_vertices > 0 and verts.shape[0] > args.max_vertices:
                idx = np.linspace(0, verts.shape[0] - 1, args.max_vertices).astype(np.int32)
                verts_use = verts[idx]
            else:
                verts_use = verts

            # Choose translation
            t = h.get("translation_opt")
            if t is None:
                continue
            t = np.asarray(t, dtype=np.float32).reshape(1, 3)

            # Project
            P = verts_use + t
            u, v, valid = project_points(K, P)
            u = u[valid]
            v = v[valid]

            # BBox from projected verts
            bbox = bbox_from_uv(u, v, W, H, margin_px=args.margin_px)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                     fill=False, linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 5), f"hand{hid} {lr}", fontsize=10)

            # Optional: overlay vertices
            if args.show_vertices:
                inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                ax.scatter(u[inb], v[inb], s=1, alpha=0.6)

        plt.tight_layout()

        if args.pause and args.pause > 0:
            plt.show(block=False)
            plt.pause(args.pause)
            plt.close(fig)
        else:
            # Show one by one; close window to advance
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
