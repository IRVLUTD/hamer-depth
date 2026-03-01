#!/usr/bin/env python3
"""
run_mediapipe_hands_on_recording_from_masks.py

Like run_hamer_on_recording_with_depth_opt_from_masks.py, but instead of HaMeR,
this script:

  1) Uses your per-frame hand segmentation masks to compute per-hand bounding boxes
     (mask values: 0=bg, 1=left, 2=right by default)
  2) Runs MediaPipe Hands on each bbox crop to estimate 21 hand joints
  3) Saves:
       - results_mediapipe.jsonl  (one line per frame, includes bboxes + joints)
       - renders/<frame>_all.jpg  (full-frame overlay: bbox + joints + skeleton)
       - renders/<frame>_hand<i>.jpg (per-hand overlay)
       - crops/<frame>_hand<i>.png (optional crops)

Folder layout (under --data_dir):
  export_pairs/
    rectified_left_frames/left_000000_t1.719302.png
    depth_frames/depth_000000_t1.719302.png       (optional; still paired and logged)
    hand_masks/left_000000_t1.719302.png          (default)

Calibration:
  calibration.json is NOT required for MediaPipe; we only store depth_K if available.

Install:
  pip install mediapipe opencv-python numpy

Usage:
  python run_mediapipe_hands_on_recording_from_masks.py ^
    --data_dir "data\\recording_20260118_101038" ^
    --out_folder "data\\recording_20260118_101038\\out_mediapipe_from_masks" ^
    --save_crops
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

import mediapipe as mp
mp_hands = mp.solutions.hands


# -----------------------------
# IO helpers
# -----------------------------
def save_jsonl_line(fp, obj: dict):
    fp.write(json.dumps(obj) + "\n")
    fp.flush()


def load_depth_meters(depth_path: Path) -> Optional[np.ndarray]:
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32)
    if np.nanmax(d) > 20.0:  # likely mm
        d *= 0.001
    return d


def load_hand_mask(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read hand mask: {mask_path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


# -----------------------------
# Mask -> bbox (with ratio padding)
# -----------------------------
def bbox_from_label_ratio(
    mask: np.ndarray,
    label: int,
    min_area: int = 50,
    pad_ratio: float = 0.20,   # 20% on each side
) -> Optional[List[float]]:
    ys, xs = np.where(mask == label)
    if xs.size == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    area = w * h
    if area < int(min_area):
        return None

    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))

    H, W = mask.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)

    return [float(x1), float(y1), float(x2), float(y2)]


# -----------------------------
# Mask-driven pairing
# -----------------------------
def build_triplets_from_masks(
    export_pairs_dir: Path,
    mask_subdir: str,
    rgb_subdir: str,
    depth_subdir: str,
    rgb_prefix: str,
    depth_prefix: str,
    ext: str,
) -> List[Tuple[Path, Optional[Path], Path]]:
    """
    Returns list of (rgb_path, depth_path_or_None, mask_path) aligned by:
      - mask filename -> rgb filename (same name)
      - depth matched by suffix after rgb_prefix; if missing, depth_path=None
    """
    mask_dir = export_pairs_dir / mask_subdir
    rgb_dir = export_pairs_dir / rgb_subdir
    depth_dir = export_pairs_dir / depth_subdir

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")
    if not depth_dir.exists():
        print(f"[warn] Depth folder not found: {depth_dir} (depth will be skipped)")
        depth_dir = None

    mask_paths = sorted(mask_dir.glob(f"*{ext}"))

    depth_map = {}
    if depth_dir is not None and depth_dir.exists():
        for dp in sorted(depth_dir.glob(f"{depth_prefix}*{ext}")):
            nm = dp.name
            if nm.startswith(depth_prefix):
                suffix = nm[len(depth_prefix):]
                depth_map[suffix] = dp

    triplets = []
    missing_rgb = 0
    missing_depth = 0

    for mpth in mask_paths:
        rp = rgb_dir / mpth.name
        if not rp.exists():
            missing_rgb += 1
            continue

        dp = None
        nm = rp.name
        if nm.startswith(rgb_prefix):
            suffix = nm[len(rgb_prefix):]
            dp = depth_map.get(suffix, None)
            if dp is None:
                missing_depth += 1
        else:
            missing_depth += 1

        triplets.append((rp, dp, mpth))

    print(f"[triplets] masks={len(mask_paths)} triplets={len(triplets)} "
          f"missing_rgb_for_mask={missing_rgb} missing_depth_for_rgb={missing_depth}")
    return triplets


# -----------------------------
# MediaPipe on bbox crop
# -----------------------------
HAND_CONNECTIONS = [
    # same topology as mediapipe HAND_CONNECTIONS, but we keep local for drawing flexibility
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

def mediapipe_hand_from_bbox(
    hands_model: mp_hands.Hands,
    img_bgr: np.ndarray,
    bbox_xyxy: List[float],
    pad_ratio_extra: float = 0.0,
    handedness_hint: Optional[str] = None,   # "Left" / "Right" / None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Runs MP Hands on a bbox crop and returns joints in FULL-image pixel coords.

    Returns:
      joints_px: (21,2) float32 (u,v) in full image coords
      joints_z:  (21,)  float32 z from MP
      info: dict
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    # clamp
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, {"reason": "invalid_bbox"}

    # optional extra padding beyond what bbox already has
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad_x = int(round(bw * pad_ratio_extra))
    pad_y = int(round(bh * pad_ratio_extra))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W - 1, x2 + pad_x)
    cy2 = min(H - 1, y2 + pad_y)

    crop_bgr = img_bgr[cy1:cy2 + 1, cx1:cx2 + 1]
    if crop_bgr.size == 0:
        return None, None, {"reason": "empty_crop"}

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = hands_model.process(crop_rgb)

    if not res.multi_hand_landmarks:
        return None, None, {"reason": "no_hand"}

    # choose best (optionally by handedness)
    best_i = 0
    best_score = -1.0
    for i in range(len(res.multi_hand_landmarks)):
        label = None
        score = 0.0
        if res.multi_handedness and i < len(res.multi_handedness):
            cls = res.multi_handedness[i].classification[0]
            label = cls.label  # "Left"/"Right"
            score = float(cls.score)
        else:
            score = 1.0

        if handedness_hint is None:
            if score > best_score:
                best_score, best_i = score, i
        else:
            # prefer matching label, fall back to score
            if label == handedness_hint and score > best_score:
                best_score, best_i = score, i
            elif best_score < 0 and score > best_score:
                best_score, best_i = score, i

    lm = res.multi_hand_landmarks[best_i].landmark
    crop_h, crop_w = crop_bgr.shape[:2]

    joints_px = np.zeros((21, 2), dtype=np.float32)
    joints_z = np.zeros((21,), dtype=np.float32)

    for k in range(21):
        u = lm[k].x * crop_w
        v = lm[k].y * crop_h
        joints_px[k, 0] = u + cx1
        joints_px[k, 1] = v + cy1
        joints_z[k] = float(lm[k].z)

    chosen_label, chosen_score = None, None
    if res.multi_handedness and best_i < len(res.multi_handedness):
        cls = res.multi_handedness[best_i].classification[0]
        chosen_label = cls.label
        chosen_score = float(cls.score)

    info = {
        "crop_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
        "handedness": chosen_label,
        "handedness_score": chosen_score,
        "reason": "ok",
    }
    return joints_px, joints_z, info


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_bbox(img_bgr: np.ndarray, bbox_xyxy: List[float], thickness: int = 2):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), thickness)


def draw_joints_and_skeleton(img_bgr: np.ndarray, joints_px: np.ndarray, radius: int = 2, thickness: int = 2):
    pts = joints_px.astype(np.int32)
    # joints
    for (u, v) in pts:
        cv2.circle(img_bgr, (int(u), int(v)), radius, (0, 255, 0), -1)
    # skeleton
    for a, b in HAND_CONNECTIONS:
        ua, va = pts[a]
        ub, vb = pts[b]
        cv2.line(img_bgr, (int(ua), int(va)), (int(ub), int(vb)), (0, 200, 0), thickness)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands on recording using hand masks -> bboxes")
    parser.add_argument("--data_dir", type=str, required=True)

    # subfolders under export_pairs
    parser.add_argument("--rgb_dirname", type=str, default="rectified_left_frames")
    parser.add_argument("--depth_dirname", type=str, default="depth_frames")
    parser.add_argument("--mask_dirname", type=str, default="hand_masks")

    # filename pairing
    parser.add_argument("--rgb_prefix", type=str, default="left_")
    parser.add_argument("--depth_prefix", type=str, default="depth_")
    parser.add_argument("--ext", type=str, default=".png")

    # mask labels + bbox knobs
    parser.add_argument("--mask_left_label", type=int, default=1)
    parser.add_argument("--mask_right_label", type=int, default=2)
    parser.add_argument("--mask_min_area", type=int, default=50)
    parser.add_argument("--mask_pad_ratio", type=float, default=0.0,
                        help="Padding ratio applied when computing bbox from mask (e.g. 0.2 = 20% each side).")
    parser.add_argument("--mp_extra_pad_ratio", type=float, default=0.0,
                        help="Extra padding ratio applied before running MediaPipe on the bbox crop.")

    # mediapipe knobs
    parser.add_argument("--mp_static_image_mode", action="store_true", default=False,
                        help="If set, uses static image mode (slower, but may detect better per-frame).")
    parser.add_argument("--mp_model_complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--mp_min_det_conf", type=float, default=0.5)
    parser.add_argument("--mp_min_track_conf", type=float, default=0.5)
    parser.add_argument("--mp_use_handedness_hint", action="store_true", default=True,
                        help="If set, use mask label (left/right) as hint to select MP detection.")

    # outputs
    parser.add_argument("--out_folder", type=str, default="out_mediapipe_from_masks")
    parser.add_argument("--save_crops", action="store_true", default=False)
    parser.add_argument("--save_per_hand_overlay", action="store_true", default=True)
    parser.add_argument("--save_full_frame_overlay", action="store_true", default=True)
    parser.add_argument("--max_pairs", type=int, default=0, help="0 = no limit")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    export_pairs_dir = data_dir / "export_pairs"

    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "renders"
    crop_dir = out_dir / "crops"
    render_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    triplets = build_triplets_from_masks(
        export_pairs_dir=export_pairs_dir,
        mask_subdir=args.mask_dirname,
        rgb_subdir=args.rgb_dirname,
        depth_subdir=args.depth_dirname,
        rgb_prefix=args.rgb_prefix,
        depth_prefix=args.depth_prefix,
        ext=args.ext,
    )
    if len(triplets) == 0:
        print("[error] no (rgb, depth, mask) triplets found.")
        return

    if args.max_pairs > 0:
        triplets = triplets[:args.max_pairs]
        print(f"[info] selecting max_pairs={args.max_pairs}.")

    jsonl_path = out_dir / "results_mediapipe.jsonl"
    print(f"[info] writing jsonl: {jsonl_path}")

    # Create ONE MediaPipe model for the whole run (important for speed)
    hands_model = mp_hands.Hands(
        static_image_mode=bool(args.mp_static_image_mode),
        model_complexity=int(args.mp_model_complexity),
        max_num_hands=2,
        min_detection_confidence=float(args.mp_min_det_conf),
        min_tracking_confidence=float(args.mp_min_track_conf),
    )

    try:
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for frame_idx, (rgb_path, depth_path, mask_path) in enumerate(triplets):
                frame_id = rgb_path.name
                if frame_id.startswith(args.rgb_prefix):
                    frame_id = frame_id[len(args.rgb_prefix):]
                frame_id = Path(frame_id).stem

                print(f"\n[frame {frame_idx}/{len(triplets)-1}] id={frame_id}")

                img_bgr = cv2.imread(str(rgb_path))
                if img_bgr is None:
                    print(f"[warn] could not read rgb: {rgb_path}, skip.")
                    continue
                H, W = img_bgr.shape[:2]

                depth = load_depth_meters(depth_path) if depth_path is not None else None

                try:
                    hand_mask = load_hand_mask(mask_path)
                except Exception as e:
                    print(f"[warn] could not read mask: {mask_path} ({e}), skip.")
                    continue

                if hand_mask.shape[0] != H or hand_mask.shape[1] != W:
                    hand_mask = cv2.resize(hand_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                # bboxes from mask labels (already padded by ratio)
                hand_bboxes: List[List[float]] = []
                hand_is_right: List[int] = []

                bbox_l = bbox_from_label_ratio(
                    hand_mask, int(args.mask_left_label),
                    min_area=int(args.mask_min_area),
                    pad_ratio=float(args.mask_pad_ratio),
                )
                if bbox_l is not None:
                    hand_bboxes.append(bbox_l)
                    hand_is_right.append(0)

                bbox_r = bbox_from_label_ratio(
                    hand_mask, int(args.mask_right_label),
                    min_area=int(args.mask_min_area),
                    pad_ratio=float(args.mask_pad_ratio),
                )
                if bbox_r is not None:
                    hand_bboxes.append(bbox_r)
                    hand_is_right.append(1)

                hands_out = []
                full_vis = img_bgr.copy()

                for hid, (bbox, is_r) in enumerate(zip(hand_bboxes, hand_is_right)):
                    handedness_hint = None
                    if args.mp_use_handedness_hint:
                        handedness_hint = "Right" if is_r == 1 else "Left"

                    joints_px, joints_z, info = mediapipe_hand_from_bbox(
                        hands_model=hands_model,
                        img_bgr=img_bgr,
                        bbox_xyxy=bbox,
                        pad_ratio_extra=float(args.mp_extra_pad_ratio),
                        handedness_hint=handedness_hint,
                    )

                    hand_rec: Dict[str, Any] = {
                        "hand_index": int(hid),
                        "is_right": int(is_r),
                        "bbox_xyxy": [float(v) for v in bbox],
                        "mp_crop_xyxy": info.get("crop_xyxy", None),
                        "mp_handedness": info.get("handedness", None),
                        "mp_handedness_score": info.get("handedness_score", None),
                        "mp_reason": info.get("reason", None),
                        "joints_px": None,
                        "joints_z": None,
                    }

                    hand_vis = img_bgr.copy()
                    draw_bbox(hand_vis, bbox)

                    if joints_px is not None:
                        draw_joints_and_skeleton(hand_vis, joints_px)
                        # also draw on full image
                        draw_bbox(full_vis, bbox)
                        draw_joints_and_skeleton(full_vis, joints_px)

                        hand_rec["joints_px"] = joints_px.astype(np.float32).tolist()  # (21,2)
                        hand_rec["joints_z"] = joints_z.astype(np.float32).tolist()    # (21,)

                    hands_out.append(hand_rec)

                    if args.save_per_hand_overlay:
                        out_hand = render_dir / f"{frame_id}_hand{hid}.jpg"
                        cv2.imwrite(str(out_hand), hand_vis)

                    if args.save_crops and info.get("crop_xyxy", None) is not None:
                        cx1, cy1, cx2, cy2 = info["crop_xyxy"]
                        crop = img_bgr[cy1:cy2 + 1, cx1:cx2 + 1]
                        if crop.size > 0:
                            cv2.imwrite(str(crop_dir / f"{frame_id}_hand{hid}.png"), crop)

                if args.save_full_frame_overlay:
                    cv2.imwrite(str(render_dir / f"{frame_id}_all.jpg"), full_vis)

                save_jsonl_line(f_jsonl, {
                    "frame_id": frame_id,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path) if depth_path is not None else None,
                    "mask_path": str(mask_path),
                    "image_size": [int(W), int(H)],
                    "depth_size": [int(depth.shape[1]), int(depth.shape[0])] if depth is not None else None,
                    "num_hands": int(len(hands_out)),
                    "hands": hands_out,
                })

                if args.debug:
                    # quick on-screen debug for this frame
                    cv2.imshow("mediapipe_full_overlay", full_vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break

    finally:
        hands_model.close()
        if args.debug:
            cv2.destroyAllWindows()

    print(f"\n[done] wrote: {jsonl_path}")
    print(f"[done] renders: {render_dir}")
    if args.save_crops:
        print(f"[done] crops: {crop_dir}")


if __name__ == "__main__":
    main()