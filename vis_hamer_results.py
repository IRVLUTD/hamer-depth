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

from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    # NOTE: we intentionally do NOT draw saved detection bboxes.
    args = ap.parse_args()

    results_path = Path(args.results)
    data_root = Path(args.data_root) if args.data_root else None

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

            mano_pose = h.get("mano_pose", None)
            if mano_pose is not None:
                mano_pose = np.asarray(mano_pose, dtype=np.float32)
                print('mano pose with length', mano_pose.shape)
            mano_shape = h.get("mano_shape", None)
            if mano_shape is not None:
                mano_shape = np.asarray(mano_shape, dtype=np.float32)
                print('mano shape with length', mano_shape.shape)

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
