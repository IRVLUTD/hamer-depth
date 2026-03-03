#!/usr/bin/env python3
"""
run_wrists_vitpose_cotracker3_online_sameframe_redetect_gated_vitpose.py

Streaming wrist tracking (ViTPose + CoTracker3-online), with:
1) Rolling window inference for cotracker3_online (window = 2*step)
2) Confidence-based reset (redetect_when either/both, patience frames)
3) SAME-FRAME reset: when reset triggers on frame t, we DO NOT commit CoTracker for t
   - we run ViTPose on frame t
   - we write ViTPose (gated) result for frame t
   - we reinitialize CoTracker on frame t (with gated wrists)
4) ViTPose outlier gating: if ViTPose wrist is too far from previous saved location,
   we reject it and keep previous saved location for that wrist.

Inputs (default):
  <recording_dir>/export_pairs/rectified_left_frames/left_*.png

Outputs:
  <out_dir>/wrist_tracks.jsonl
  <out_dir>/wrist_tracks.csv
  <out_dir>/renders/<frame>_overlay.jpg   (optional)
  <out_dir>/wrist_tracks_overlay.mp4      (optional)

Deps:
  pip install opencv-python numpy torch pillow
  - vitpose_model.ViTPoseModel
  - torch.hub facebookresearch/co-tracker (cotracker3_online)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

try:
    import cv2
except Exception:
    cv2 = None

from vitpose_model import ViTPoseModel


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# COCO-17 indices
LEFT_WRIST = 9
RIGHT_WRIST = 10


# -------------------------
# IO
# -------------------------
def list_images(images_dir: Path, prefix: str = "") -> List[Path]:
    paths: List[Path] = []
    for ext in IMG_EXTS:
        if prefix:
            paths.extend(images_dir.glob(f"{prefix}*{ext}"))
        else:
            paths.extend(images_dir.glob(f"*{ext}"))
    return sorted(paths)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_jsonl_line(fp, obj: dict) -> None:
    fp.write(json.dumps(obj) + "\n")
    fp.flush()


# -------------------------
# Viz
# -------------------------
def draw_points_overlay(
    img_bgr: np.ndarray,
    left_xy: Optional[Tuple[float, float]],
    right_xy: Optional[Tuple[float, float]],
    left_conf: float,
    right_conf: float,
    left_vis: bool,
    right_vis: bool,
    source: str,
) -> np.ndarray:
    if cv2 is None:
        return img_bgr
    out = img_bgr.copy()
    H, W = out.shape[:2]

    def _draw_one(xy, conf, vis, label):
        if xy is None:
            return
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            # green if visible, red otherwise
            color = (0, 255, 0) if vis else (0, 0, 255)
            cv2.circle(out, (xi, yi), 7, color, -1)
            cv2.putText(out, f"{label} c={conf:.2f}", (xi + 10, yi - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    _draw_one(left_xy, left_conf, left_vis, "L")
    _draw_one(right_xy, right_conf, right_vis, "R")

    cv2.putText(out, f"source: {source}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def write_overlay_video_from_records(
    out_mp4: Path,
    image_paths: List[Path],
    records: List[Dict[str, Any]],
    fps: int,
) -> None:
    if cv2 is None:
        print("[WARN] OpenCV not available; skipping overlay video.")
        return
    if not image_paths:
        return

    im0 = cv2.imread(str(image_paths[0]))
    if im0 is None:
        print("[WARN] Could not read first frame; skipping overlay video.")
        return
    H, W = im0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (W, H))

    by_name = {r["image_name"]: r for r in records if "image_name" in r}

    for p in image_paths:
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros((H, W, 3), dtype=np.uint8)

        r = by_name.get(p.name, {})
        lxy = r.get("left_wrist_xy")
        rxy = r.get("right_wrist_xy")

        left_xy = tuple(lxy) if isinstance(lxy, list) and len(lxy) == 2 else None
        right_xy = tuple(rxy) if isinstance(rxy, list) and len(rxy) == 2 else None

        left_conf = float(r.get("left_conf", 0.0))
        right_conf = float(r.get("right_conf", 0.0))

        left_vis = bool(r.get("left_visible", False))
        right_vis = bool(r.get("right_visible", False))
        source = str(r.get("source", ""))

        vis = draw_points_overlay(im, left_xy, right_xy, left_conf, right_conf, left_vis, right_vis, source)
        vw.write(vis)

    vw.release()


# -------------------------
# ViTPose
# -------------------------
def vitpose_detect_wrists_fullframe(
    cpm: ViTPoseModel,
    img_bgr: np.ndarray,
    box_score: float,
    min_kpt_conf: float,
) -> Tuple[Optional[Tuple[float, float]], float, Optional[Tuple[float, float]], float]:
    """
    Returns:
      left_xy, left_conf, right_xy, right_conf
    """
    H, W = img_bgr.shape[:2]
    box = np.array([[0.0, 0.0, float(W - 1), float(H - 1), float(box_score)]], dtype=np.float32)
    img_rgb = img_bgr[:, :, ::-1].copy()

    out = cpm.predict_pose(img_rgb, [box])
    if not out or len(out) == 0 or "keypoints" not in out[0]:
        return None, 0.0, None, 0.0

    kps = np.asarray(out[0]["keypoints"], dtype=np.float32)

    left_xy = None
    right_xy = None
    left_c = 0.0
    right_c = 0.0

    if kps.shape[0] > LEFT_WRIST:
        x, y, c = float(kps[LEFT_WRIST, 0]), float(kps[LEFT_WRIST, 1]), float(kps[LEFT_WRIST, 2])
        if np.isfinite(x) and np.isfinite(y) and c >= min_kpt_conf:
            left_xy, left_c = (x, y), c

    if kps.shape[0] > RIGHT_WRIST:
        x, y, c = float(kps[RIGHT_WRIST, 0]), float(kps[RIGHT_WRIST, 1]), float(kps[RIGHT_WRIST, 2])
        if np.isfinite(x) and np.isfinite(y) and c >= min_kpt_conf:
            right_xy, right_c = (x, y), c

    return left_xy, left_c, right_xy, right_c


def gate_vitpose_xy(
    vit_xy: Optional[Tuple[float, float]],
    vit_c: float,
    prev_xy: Optional[Tuple[float, float]],
    max_jump_px: float,
    min_conf: float,
) -> Tuple[Optional[Tuple[float, float]], float, bool]:
    """
    Returns (xy_used, conf_used, accepted).
    If vitpose is missing/low-conf or too far from prev, fall back to prev.
    """
    if vit_xy is None or not np.isfinite(vit_c) or float(vit_c) < float(min_conf):
        if prev_xy is not None:
            return prev_xy, 0.0, False
        return None, 0.0, False

    if prev_xy is not None:
        d = float(np.linalg.norm(np.array(vit_xy, dtype=np.float32) - np.array(prev_xy, dtype=np.float32)))
        if d > float(max_jump_px):
            return prev_xy, 0.0, False

    return vit_xy, float(vit_c), True


# -------------------------
# CoTracker streaming (windowed) with confidence
# -------------------------
@dataclass
class TrackerState:
    tracker: Any
    initialized: bool
    H: int
    W: int
    buffer: deque  # stores (1,1,3,H,W) tensors


def load_frame_tensor_1(
    frame_path: Path,
    device: torch.device,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      frame_t: (1,1,3,H,W) float in [0,1]
      H, W
    """
    im = Image.open(frame_path).convert("RGB")
    if target_hw is not None:
        th, tw = target_hw
        if im.size != (tw, th):
            im = im.resize((tw, th), resample=Image.BILINEAR)

    W, H = im.size
    arr = np.array(im, dtype=np.uint8, copy=True)  # writable copy (avoid torch warning)
    t = torch.from_numpy(arr).to(device).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).contiguous()  # (1,1,3,H,W)
    return t, H, W


def init_cotracker(
    tracker_state: TrackerState,
    frame_t: torch.Tensor,
    left_xy: Optional[Tuple[float, float]],
    right_xy: Optional[Tuple[float, float]],
) -> None:
    """
    Initialize online tracker on a single frame with queries. Also resets rolling buffer.
    """
    _, _, _, H, W = frame_t.shape

    pts = []
    pts.append((0.0, 0.0) if left_xy is None else (float(left_xy[0]), float(left_xy[1])))
    pts.append((0.0, 0.0) if right_xy is None else (float(right_xy[0]), float(right_xy[1])))

    pts = [(float(np.clip(x, 0.0, W - 1.0)), float(np.clip(y, 0.0, H - 1.0))) for (x, y) in pts]

    queries_txy = np.array([
        [0.0, pts[0][0], pts[0][1]],
        [0.0, pts[1][0], pts[1][1]],
    ], dtype=np.float32)
    queries = torch.from_numpy(queries_txy).to(frame_t.device).unsqueeze(0)  # (1,2,3)

    tracker_state.buffer.clear()
    tracker_state.buffer.append(frame_t)

    tracker_state.tracker(
        video_chunk=frame_t,
        is_first_step=True,
        queries=queries,
        grid_size=0,
        add_support_grid=False,
    )

    tracker_state.initialized = True
    tracker_state.H = H
    tracker_state.W = W


def step_cotracker_windowed_conf(
    tracker_state: TrackerState,
    frame_t: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns prediction for LAST frame in window:
      xy_last:   (2,2) float32
      conf_last: (2,) float32 (if model returns bool, conf is 0/1)
    """
    step = int(getattr(tracker_state.tracker, "step", 8))
    win = 2 * step

    tracker_state.buffer.append(frame_t)

    buf = list(tracker_state.buffer)
    if len(buf) < win:
        buf = [buf[0]] * (win - len(buf)) + buf
    else:
        buf = buf[-win:]

    video_chunk = torch.cat(buf, dim=1)  # (1,win,3,H,W)

    with torch.no_grad():
        pred_tracks, pred_vis = tracker_state.tracker(video_chunk=video_chunk)

    tr = pred_tracks.detach().float().cpu().numpy()[0]  # (win,2,2)

    vi = pred_vis.detach().cpu().numpy()[0]
    if vi.ndim == 3 and vi.shape[-1] == 1:
        vi = vi[..., 0]
    conf = vi.astype(np.float32)

    return tr[-1].astype(np.float32), conf[-1].astype(np.float32)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording_dir", type=str, required=True)
    ap.add_argument("--images_rel", type=str, default=r"export_pairs\rectified_left_frames")
    ap.add_argument("--rgb_prefix", type=str, default="left_")
    ap.add_argument("--out_rel", type=str, default=r"export_pairs\wrist_vitpose_cotracker3")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps", type=int, default=30)

    # ViTPose
    ap.add_argument("--vitpose_min_kpt_conf", type=float, default=0.3)
    ap.add_argument("--vitpose_box_score", type=float, default=1.0)
    ap.add_argument("--max_init_search", type=int, default=30)
    ap.add_argument("--vitpose_max_jump_px", type=float, default=80.0,
                    help="Reject ViTPose wrist if too far from previous saved location.")

    # Redetect logic (CoTracker)
    ap.add_argument("--vis_thresh", type=float, default=0.6)
    ap.add_argument("--occlusion_patience", type=int, default=1)
    ap.add_argument("--redetect_when", choices=["either", "both"], default="either")
    ap.add_argument("--min_gap_between_resets", type=int, default=0,
                    help="0 recommended when using immediate re-detect. Increase to reduce reset spam.")
    ap.add_argument("--force_reset_below", type=float, default=0.05,
                    help="Ignore min_gap and force reset if any wrist conf is below this (e.g., 0.0).")
    ap.add_argument(
        "--cotracker_max_jump_px",
        type=float,
        default=40.0,
        help="If a wrist moves more than this many pixels from previous saved location in 1 frame, treat as bad and reset.",
    )                    

    # Logging / outputs
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--write_renders", action="store_true")
    ap.add_argument("--write_video", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    recording_dir = Path(args.recording_dir)
    images_dir = recording_dir / Path(args.images_rel)
    out_dir = recording_dir / Path(args.out_rel)
    ensure_dir(out_dir)

    render_dir = out_dir / "renders"
    if args.write_renders:
        ensure_dir(render_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    image_paths = list_images(images_dir, prefix=args.rgb_prefix)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] frames: {len(image_paths)}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] vis_thresh={args.vis_thresh} patience={args.occlusion_patience} redetect_when={args.redetect_when}")
    print(f"[INFO] vitpose_max_jump_px={args.vitpose_max_jump_px} vitpose_min_kpt_conf={args.vitpose_min_kpt_conf}")
    print(f"[INFO] min_gap_between_resets={args.min_gap_between_resets} force_reset_below={args.force_reset_below}")

    # Models
    cpm = ViTPoseModel(str(device))

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    cotracker.eval()
    step = int(getattr(cotracker, "step", 8))
    tracker_state = TrackerState(
        tracker=cotracker,
        initialized=False,
        H=-1,
        W=-1,
        buffer=deque(maxlen=2 * step),
    )
    print(f"[INFO] CoTracker step={step} window={2*step}")

    def read_bgr(p: Path) -> Optional[np.ndarray]:
        if cv2 is not None:
            return cv2.imread(str(p))
        try:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            return rgb[:, :, ::-1].copy()
        except Exception:
            return None

    # -------------------------
    # Initial wrists from ViTPose
    # -------------------------
    init_idx = 0
    init_left = None
    init_right = None
    init_lc = 0.0
    init_rc = 0.0

    search_to = min(len(image_paths), int(args.max_init_search) if int(args.max_init_search) > 0 else 1)
    for i in range(search_to):
        img_bgr = read_bgr(image_paths[i])
        if img_bgr is None:
            continue
        lxy, lc, rxy, rc = vitpose_detect_wrists_fullframe(
            cpm, img_bgr,
            box_score=float(args.vitpose_box_score),
            min_kpt_conf=float(args.vitpose_min_kpt_conf),
        )
        if lxy is not None or rxy is not None:
            init_idx = i
            init_left, init_right = lxy, rxy
            init_lc, init_rc = lc, rc
            print(f"[INFO] vitpose init @ {i}: left={init_left} (c={lc:.2f}) right={init_right} (c={rc:.2f})")
            break

    if init_left is None and init_right is None:
        raise RuntimeError("ViTPose could not find any wrist in the initial search window.")

    # Init tracker
    frame_t, _, _ = load_frame_tensor_1(image_paths[init_idx], device=device, target_hw=None)
    init_cotracker(tracker_state, frame_t, init_left, init_right)

    # Outputs
    T = len(image_paths)
    left_xy_all = np.full((T, 2), np.nan, dtype=np.float32)
    right_xy_all = np.full((T, 2), np.nan, dtype=np.float32)
    left_vis_all = np.zeros((T,), dtype=bool)
    right_vis_all = np.zeros((T,), dtype=bool)
    left_conf_all = np.zeros((T,), dtype=np.float32)
    right_conf_all = np.zeros((T,), dtype=np.float32)
    source_all = [""] * T

    # Record init frame (ViTPose)
    if init_left is not None:
        left_xy_all[init_idx] = (init_left[0], init_left[1])
        left_vis_all[init_idx] = True
        left_conf_all[init_idx] = float(init_lc)
    if init_right is not None:
        right_xy_all[init_idx] = (init_right[0], init_right[1])
        right_vis_all[init_idx] = True
        right_conf_all[init_idx] = float(init_rc)
    source_all[init_idx] = "vitpose_init"

    patience = max(1, int(args.occlusion_patience))
    min_gap = max(0, int(args.min_gap_between_resets))
    streak = 0
    last_reset = -10**9

    # -------------------------
    # Stream frames
    # -------------------------
    for t in range(init_idx + 1, T):
        target_hw = (tracker_state.H, tracker_state.W) if tracker_state.H > 0 else None
        frame_t, _, _ = load_frame_tensor_1(image_paths[t], device=device, target_hw=target_hw)

        # Predict with CoTracker (do NOT commit yet)
        xy_pred, conf = step_cotracker_windowed_conf(tracker_state, frame_t)

        # previous saved locations (t-1) for jump gating
        prev_l = None
        prev_r = None
        if t - 1 >= 0 and np.all(np.isfinite(left_xy_all[t - 1])):
            prev_l = left_xy_all[t - 1].astype(np.float32)
        if t - 1 >= 0 and np.all(np.isfinite(right_xy_all[t - 1])):
            prev_r = right_xy_all[t - 1].astype(np.float32)

        jump_bad_l = False
        jump_bad_r = False
        if prev_l is not None and np.all(np.isfinite(xy_pred[0])):
            jump_bad_l = (np.linalg.norm(xy_pred[0] - prev_l) > float(args.cotracker_max_jump_px))
        if prev_r is not None and np.all(np.isfinite(xy_pred[1])):
            jump_bad_r = (np.linalg.norm(xy_pred[1] - prev_r) > float(args.cotracker_max_jump_px))

        lx, ly = float(xy_pred[0, 0]), float(xy_pred[0, 1])
        rx, ry = float(xy_pred[1, 0]), float(xy_pred[1, 1])
        lconf = float(conf[0]); rconf = float(conf[1])

        low_l = (lconf < float(args.vis_thresh)) or jump_bad_l
        low_r = (rconf < float(args.vis_thresh)) or jump_bad_r

        if args.redetect_when == "either":
            low = (low_l or low_r)
        else:
            low = (low_l and low_r)

        # Decide whether we should ignore min_gap (very low confidence)
        force = (lconf < float(args.force_reset_below)) or (rconf < float(args.force_reset_below))

        # Update streak (min_gap blocks unless forced)
        if (t - last_reset) < min_gap and (not force):
            streak = 0
        else:
            streak = (streak + 1) if low else 0

        do_reset = (streak >= patience)

        # Logging
        if args.verbose:
            print(
                f"[FRAME {t:05d}/{T}] "
                f"L=({lx:.1f},{ly:.1f}) c={lconf:.2f} low={low_l} | "
                f"R=({rx:.1f},{ry:.1f}) c={rconf:.2f} low={low_r} | "
                f"streak={streak} reset={do_reset} gap={t-last_reset} force={force}"
            )
        elif (t % int(args.log_every) == 0):
            print(f"[PROGRESS] frame {t}/{T}  streak={streak}")

        if do_reset:
            # SAME-FRAME ViTPose re-detect, and DO NOT save CoTracker result for this frame
            img_bgr = read_bgr(image_paths[t])
            if img_bgr is not None:
                lxy_raw, lc_raw, rxy_raw, rc_raw = vitpose_detect_wrists_fullframe(
                    cpm, img_bgr,
                    box_score=float(args.vitpose_box_score),
                    min_kpt_conf=float(args.vitpose_min_kpt_conf),
                )

                # Previous saved locations (t-1)
                prev_l = None
                prev_r = None
                if t - 1 >= 0 and np.all(np.isfinite(left_xy_all[t - 1])):
                    prev_l = (float(left_xy_all[t - 1, 0]), float(left_xy_all[t - 1, 1]))
                if t - 1 >= 0 and np.all(np.isfinite(right_xy_all[t - 1])):
                    prev_r = (float(right_xy_all[t - 1, 0]), float(right_xy_all[t - 1, 1]))

                # Gate ViTPose outliers
                lxy, lc, l_acc = gate_vitpose_xy(
                    lxy_raw, lc_raw, prev_l,
                    max_jump_px=float(args.vitpose_max_jump_px),
                    min_conf=float(args.vitpose_min_kpt_conf),
                )
                rxy, rc, r_acc = gate_vitpose_xy(
                    rxy_raw, rc_raw, prev_r,
                    max_jump_px=float(args.vitpose_max_jump_px),
                    min_conf=float(args.vitpose_min_kpt_conf),
                )

                # Save gated ViTPose for THIS frame
                if lxy is not None:
                    left_xy_all[t] = (lxy[0], lxy[1])
                    left_vis_all[t] = True
                    left_conf_all[t] = float(lc)
                else:
                    left_vis_all[t] = False
                    left_conf_all[t] = 0.0

                if rxy is not None:
                    right_xy_all[t] = (rxy[0], rxy[1])
                    right_vis_all[t] = True
                    right_conf_all[t] = float(rc)
                else:
                    right_vis_all[t] = False
                    right_conf_all[t] = 0.0

                source_all[t] = "vitpose_reset_same_frame_gated"

                # Re-init tracker on THIS same frame (using gated wrists)
                init_cotracker(tracker_state, frame_t, lxy, rxy)

                print(
                    f"\n[RESET @ frame {t}] streak={streak} "
                    f"cotracker_conf(L={lconf:.2f}, R={rconf:.2f}) "
                    f"vit_raw(L={lxy_raw},c={lc_raw:.2f} | R={rxy_raw},c={rc_raw:.2f}) "
                    f"vit_used(L={lxy},acc={l_acc} | R={rxy},acc={r_acc})\n"
                )

                last_reset = t
                streak = 0
            else:
                # If can't read image, fall back to saving CoTracker
                if np.isfinite(lx) and np.isfinite(ly):
                    left_xy_all[t] = (lx, ly)
                if np.isfinite(rx) and np.isfinite(ry):
                    right_xy_all[t] = (rx, ry)
                left_conf_all[t] = lconf
                right_conf_all[t] = rconf
                left_vis_all[t] = not low_l
                right_vis_all[t] = not low_r
                source_all[t] = "cotracker_fallback_readfail"
        else:
            # Commit CoTracker for this frame
            if np.isfinite(lx) and np.isfinite(ly):
                left_xy_all[t] = (lx, ly)
            if np.isfinite(rx) and np.isfinite(ry):
                right_xy_all[t] = (rx, ry)

            left_conf_all[t] = lconf
            right_conf_all[t] = rconf
            left_vis_all[t] = not low_l
            right_vis_all[t] = not low_r
            source_all[t] = "cotracker"

        # Optional render/debug (uses saved values)
        if args.write_renders or args.debug:
            if cv2 is not None:
                im = cv2.imread(str(image_paths[t]))
                if im is not None:
                    lxy_s = None if not np.all(np.isfinite(left_xy_all[t])) else (float(left_xy_all[t, 0]), float(left_xy_all[t, 1]))
                    rxy_s = None if not np.all(np.isfinite(right_xy_all[t])) else (float(right_xy_all[t, 0]), float(right_xy_all[t, 1]))
                    vis_im = draw_points_overlay(
                        im,
                        lxy_s,
                        rxy_s,
                        float(left_conf_all[t]),
                        float(right_conf_all[t]),
                        bool(left_vis_all[t]),
                        bool(right_vis_all[t]),
                        source_all[t],
                    )

                    if args.write_renders:
                        stem = Path(image_paths[t].name).stem
                        cv2.imwrite(str(render_dir / f"{stem}_overlay.jpg"), vis_im)

                    if args.debug:
                        cv2.imshow("wrist tracking (ESC to stop)", vis_im)
                        if (cv2.waitKey(1) & 0xFF) == 27:
                            break

    if args.debug and cv2 is not None:
        cv2.destroyAllWindows()

    # -------------------------
    # Write outputs
    # -------------------------
    jsonl_path = out_dir / "wrist_tracks.jsonl"
    csv_path = out_dir / "wrist_tracks.csv"

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(image_paths):
            lxy = left_xy_all[i]
            rxy = right_xy_all[i]
            rec = {
                "frame_index": i,
                "image_name": p.name,
                "image_path": str(p),
                "left_wrist_xy": None if not np.all(np.isfinite(lxy)) else [float(lxy[0]), float(lxy[1])],
                "right_wrist_xy": None if not np.all(np.isfinite(rxy)) else [float(rxy[0]), float(rxy[1])],
                "left_visible": bool(left_vis_all[i]),
                "right_visible": bool(right_vis_all[i]),
                "left_conf": float(left_conf_all[i]),
                "right_conf": float(right_conf_all[i]),
                "source": source_all[i],
            }
            save_jsonl_line(f, rec)
            records.append(rec)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index", "image_name",
            "left_x", "left_y", "left_conf", "left_vis",
            "right_x", "right_y", "right_conf", "right_vis",
            "source",
        ])
        for i, p in enumerate(image_paths):
            lxy = left_xy_all[i]
            rxy = right_xy_all[i]
            w.writerow([
                i, p.name,
                "" if not np.all(np.isfinite(lxy)) else f"{float(lxy[0]):.3f}",
                "" if not np.all(np.isfinite(lxy)) else f"{float(lxy[1]):.3f}",
                f"{float(left_conf_all[i]):.3f}",
                int(left_vis_all[i]),
                "" if not np.all(np.isfinite(rxy)) else f"{float(rxy[0]):.3f}",
                "" if not np.all(np.isfinite(rxy)) else f"{float(rxy[1]):.3f}",
                f"{float(right_conf_all[i]):.3f}",
                int(right_vis_all[i]),
                source_all[i],
            ])

    print(f"[DONE] wrote: {jsonl_path}")
    print(f"[DONE] wrote: {csv_path}")

    if args.write_video:
        out_mp4 = out_dir / "wrist_tracks_overlay.mp4"
        write_overlay_video_from_records(out_mp4, image_paths, records, fps=int(args.fps))
        print(f"[DONE] wrote: {out_mp4}")


if __name__ == "__main__":
    main()