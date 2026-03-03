#!/usr/bin/env python3
"""
run_body_pose_fullframe_box.py

Ego-centric shortcut:
  - Use the ENTIRE image as a single "person" box
  - Run ViTPoseModel keypoint estimation
  - Render keypoints/skeleton on the full image

Inputs:
  data_dir/export_pairs/rectified_left_frames/left_*.png

Outputs:
  out_folder/results_bodypose_fullframe.jsonl
  out_folder/renders/<frame>_pose.jpg

Install:
  pip install opencv-python numpy torch
(Plus whatever your ViTPoseModel needs.)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import List, Dict, Any

import cv2
import numpy as np
import torch

from vitpose_model import ViTPoseModel


def save_jsonl_line(fp, obj: dict):
    fp.write(json.dumps(obj) + "\n")
    fp.flush()


def list_rgb_frames(export_pairs_dir: Path, rgb_subdir: str, rgb_prefix: str, ext: str) -> List[Path]:
    rgb_dir = export_pairs_dir / rgb_subdir
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
    paths = sorted(rgb_dir.glob(f"{rgb_prefix}*{ext}"))
    if len(paths) == 0:
        paths = sorted(rgb_dir.glob(f"*{ext}"))
    print(f"[rgb] found {len(paths)} frames in {rgb_dir}")
    return paths


# COCO-17 skeleton edges (draw on first 17 keypoints if available)
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# COCO-17 indices
LEFT_ELBOW  = 7
RIGHT_ELBOW = 8
LEFT_WRIST  = 9
RIGHT_WRIST = 10


def draw_keypoints(img_bgr: np.ndarray, kps: np.ndarray, min_conf: float = 0.3):
    """
    kps: (K,3) [x,y,conf]
    """
    for i in range(kps.shape[0]):
        x, y, c = float(kps[i, 0]), float(kps[i, 1]), float(kps[i, 2])
        if c < min_conf:
            continue

        # ---- Special coloring ----
        if i in [LEFT_WRIST, RIGHT_WRIST]:
            color = (0, 0, 255)        # RED for wrists
            radius = 6
        elif i in [LEFT_ELBOW, RIGHT_ELBOW]:
            color = (0, 255, 255)      # YELLOW for elbows
            radius = 5
        else:
            color = (0, 255, 0)        # GREEN for others
            radius = 3

        cv2.circle(img_bgr, (int(round(x)), int(round(y))), radius, color, -1)


def draw_skeleton(img_bgr: np.ndarray, kps: np.ndarray, edges, min_conf: float = 0.3):
    for a, b in edges:
        if a >= kps.shape[0] or b >= kps.shape[0]:
            continue
        xa, ya, ca = float(kps[a, 0]), float(kps[a, 1]), float(kps[a, 2])
        xb, yb, cb = float(kps[b, 0]), float(kps[b, 1]), float(kps[b, 2])
        if ca < min_conf or cb < min_conf:
            continue
        cv2.line(img_bgr, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (0, 200, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="ViTPose on full-frame box (no person detector)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--rgb_dirname", type=str, default="rectified_left_frames")
    parser.add_argument("--rgb_prefix", type=str, default="left_")
    parser.add_argument("--ext", type=str, default=".png")

    parser.add_argument("--out_folder", type=str, default="out_body_pose_fullframe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_kpt_conf", type=float, default=0.3)
    parser.add_argument("--box_score", type=float, default=1.0, help="Dummy score appended to the box (x1,y1,x2,y2,score)")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    export_pairs_dir = data_dir / "export_pairs"

    out_dir = export_pairs_dir / Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = list_rgb_frames(export_pairs_dir, args.rgb_dirname, args.rgb_prefix, args.ext)
    if args.max_frames > 0:
        rgb_paths = rgb_paths[:args.max_frames]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; using cpu")
        device = "cpu"

    cpm = ViTPoseModel(device)

    jsonl_path = out_dir / "results_bodypose_fullframe.jsonl"
    print(f"[info] writing jsonl: {jsonl_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, rgb_path in enumerate(rgb_paths):
            frame_id = rgb_path.name
            print(frame_id)
            if frame_id.startswith(args.rgb_prefix):
                frame_id = frame_id[len(args.rgb_prefix):]
            frame_id = Path(frame_id).stem

            img_bgr = cv2.imread(str(rgb_path))
            if img_bgr is None:
                print(f"[warn] could not read: {rgb_path}")
                continue
            H, W = img_bgr.shape[:2]

            # single full-frame box (x1,y1,x2,y2,score)
            box = np.array([[0.0, 0.0, float(W - 1), float(H - 1), float(args.box_score)]], dtype=np.float32)

            img_rgb = img_bgr[:, :, ::-1].copy()

            # ViTPoseModel API from your previous script: predict_pose(img_rgb, [boxes5])
            out = cpm.predict_pose(img_rgb, [box])

            rec: Dict[str, Any] = {
                "frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "image_size": [int(W), int(H)],
                "box_xyxy": [0.0, 0.0, float(W - 1), float(H - 1)],
                "keypoints": None,
            }

            vis = img_bgr.copy()

            # out is usually list of persons; here we passed 1 box so expect 1 result
            if out and len(out) > 0 and "keypoints" in out[0]:
                kps = np.asarray(out[0]["keypoints"], dtype=np.float32)  # (K,3)
                rec["keypoints"] = kps.tolist()

                kps_body = kps[:17] if kps.shape[0] >= 17 else kps
                draw_skeleton(vis, kps_body, COCO17_EDGES, min_conf=float(args.min_kpt_conf))
                draw_keypoints(vis, kps_body, min_conf=float(args.min_kpt_conf))

            save_jsonl_line(f, rec)
            cv2.imwrite(str(render_dir / f"{frame_id}_pose.jpg"), vis)

            if args.debug:
                cv2.imshow("vitpose full-frame box (ESC to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    if args.debug:
        cv2.destroyAllWindows()

    print(f"[done] wrote: {jsonl_path}")
    print(f"[done] renders: {render_dir}")


if __name__ == "__main__":
    main()