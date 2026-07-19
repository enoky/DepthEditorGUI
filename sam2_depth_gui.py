#!/usr/bin/env python3
"""
sam2_depth_gui.py -- local browser GUI for masking objects in a depth video.
============================================================================

Workflow (top to bottom in the UI):
  1. Load the RGB video and the matching depth-map video (e.g. DepthCrafter).
     Optional "depth ghost": blends a colormapped copy of the depth video
     into the frames SAM2 sees (depth is resized to the RGB frame size), so
     depth boundaries show up as color edges and prompts/tracking can follow
     them. 0 = off; ~30-50% is a good range.
  2. Scrub to any clear frame and click each object to mask (or use the
     "box" click mode). Each object gets its own ID and color, and
     the mask preview appears instantly. Red-cross points fix
     over-segmentation.
  3. "Track objects" runs SAM2 video propagation forwards AND backwards from
     the prompt frames. Scrub (or use the arrow keys / preview clip) to
     verify; add corrective points where it drifts and track again.
     Objects that cross paths are kept mutually exclusive (overlapping
     pixels are split by nearest-object / motion continuity after each
     run); "erase (box)" cuts mistakes out of one object's mask on one
     frame, "erase (click)" is a single-frame background (-) click (SAM2
     carves the clicked region out of the tracked mask without a
     re-track), and both re-apply after every re-track.
  4. Switch the view to "Depth before | after", pick a mode (brightness /
     compress / inpaint) and tune sliders with instant single-frame preview. Settings
     can be overridden per object; the histogram helps pick thresholds.
     "Snap to depth" grows each mask into adjacent pixels of similar depth
     value, so masks conform to the depth object's silhouette even where
     DepthCrafter's blobs overhang the RGB outline.
  5. "Render" writes the processed depth MP4 (never overwrites existing
     files; can also export a mask matte video for external compositing).

Extras: save/load the whole project (prompts + masks + settings) to a .npz,
cancel long operations, test-render a short range before committing.
10-bit depth videos (e.g. DepthCrafter HEVC output) are preserved: frames are
decoded via ffmpeg at 16-bit, processed in 16-bit, and rendered as 10-bit
HEVC (libx265). 8-bit sources render as 8-bit H.264 exactly as before.

Keyboard: Left/Right = 1 frame, Shift+Left/Right = 10 frames.

Install (Windows, in a terminal):
  py -m pip install gradio opencv-python numpy
  # PyTorch with CUDA (pick your CUDA version at pytorch.org if different):
  py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  # SAM2 (needs Git for Windows: winget install Git.Git):
  py -m pip install "git+https://github.com/facebookresearch/sam2.git"
  # optional but recommended for best encode quality:
  winget install Gyan.FFmpeg

Checkpoint (~900 MB) -- put it in .\\checkpoints\\ :
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

Run:
  py sam2_depth_gui.py
A browser tab opens at http://127.0.0.1:7860 (everything stays local).
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# ----------------------------------------------------------------------------
# Persistent config (last used paths + settings)
# ----------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).with_name("s2dm_config.json")

DEFAULTS = {
    "mode": "brightness", "brightness": 0.5, "threshold": 140, "gain": 0.35,
    "inpaint_radius": 5, "dilate_px": 2, "feather_px": 4,
    "snap_px": 0, "snap_tol": 12,
}
SET_KEYS = tuple(DEFAULTS)  # order matters: must match setting_comps below


def _migrate_settings(st: dict) -> dict:
    """Accept settings saved before 'dim' became the 'brightness' mode."""
    st = dict(st)
    if st.get("mode") == "dim":
        st["mode"] = "brightness"
    if "dim_factor" in st:
        st.setdefault("brightness", st.pop("dim_factor"))
    return st

DEFAULT_SCOPE = "default (all objects)"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


CFG = _load_config()


def save_config(**updates) -> None:
    CFG.update({k: v for k, v in updates.items() if v is not None})
    try:
        CONFIG_PATH.write_text(json.dumps(CFG, indent=2), encoding="utf-8")
    except OSError:
        pass


def _cfg_file(key: str) -> str | None:
    p = CFG.get(key)
    return p if p and Path(p).exists() else None


# ----------------------------------------------------------------------------
# Session (single local user)
# ----------------------------------------------------------------------------

S = {
    "frames_dir": None,      # Path to extracted RGB jpgs
    "n_frames": 0,           # usable frames = min(rgb, depth)
    "rgb_size": (0, 0),      # (w, h)
    "depth_path": None,
    "depth_size": (0, 0),
    "depth_fps": 24.0,
    "depth_bits": 8,         # source bit depth (10 for DepthCrafter HEVC)
    "ghost": 0.0,            # depth-ghost blend fraction baked into frames
    "depth_cap": None,       # persistent cv2.VideoCapture for seeking
    "prompts": [],           # [{kind:"point",obj,frame,x,y,label} |
                             #  {kind:"box",obj,frame,x1,y1,x2,y2}]
    "masks": {},             # obj_id -> {frame_idx -> packed bits (depth res)}
    "settings": {**DEFAULTS,
                 **{k: v
                    for k, v in _migrate_settings(CFG.get("settings",
                                                          {})).items()
                    if k in DEFAULTS}},
    "obj_settings": {},      # obj_id -> settings override dict
    "pending_box": None,     # first corner of an in-progress box prompt
    "sel_row": None,         # selected row in the prompts table
    "cancel": False,
    "preview_failed": False,  # SAM2 unavailable -> skip instant previews
    "state_tracked": False,   # SAM2 state has been propagated at least once
    "predictor": None,
    "state": None,
    "device": None,
    "log": [],
}

PALETTE = [  # RGB, one per object ID (cycles)
    (255, 200, 0), (0, 200, 255), (120, 255, 120), (255, 120, 255),
    (255, 140, 60), (140, 160, 255), (255, 255, 120), (0, 255, 180),
    (255, 90, 120), (190, 130, 255),
]

# The depth ghost blends TURBO (blue-cyan-green-yellow-orange-red) into the
# frames, so those hues stop reading as "mask". TURBO never produces
# magenta/violet/pink or pure white -- use those when a ghost is active.
GHOST_PALETTE = [
    (255, 60, 255), (255, 255, 255), (165, 80, 255), (255, 0, 150),
    (230, 190, 255), (200, 0, 255), (255, 150, 220), (216, 100, 255),
    (250, 235, 255), (190, 40, 190),
]


def _color(obj: int) -> tuple[int, int, int]:
    pal = GHOST_PALETTE if S.get("ghost") else PALETTE
    return pal[(int(obj) - 1) % len(pal)]


def log(msg: str) -> None:
    S["log"].append(f"`{time.strftime('%H:%M:%S')}` {msg}")
    del S["log"][:-200]


def log_md() -> str:
    if not S["log"]:
        return "Load the two videos to begin."
    return "\n".join(f"- {m}" for m in reversed(S["log"][-8:]))


def require_loaded() -> None:
    if S["frames_dir"] is None:
        raise gr.Error("Load videos first.")


# ----------------------------------------------------------------------------
# Mask storage helpers (packed bits per object per frame, at depth resolution)
# ----------------------------------------------------------------------------


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8).ravel())


def _unpack(obj: int, idx: int) -> np.ndarray | None:
    packed = S["masks"].get(obj, {}).get(idx)
    if packed is None:
        return None
    w, h = S["depth_size"]
    return np.unpackbits(packed, count=w * h).reshape(h, w).astype(bool)


def union_mask(idx: int) -> np.ndarray | None:
    u = None
    for obj in S["masks"]:
        m = _unpack(obj, idx)
        if m is not None:
            u = m if u is None else (u | m)
    return u


def settings_for(obj: int) -> dict:
    return {**S["settings"], **S["obj_settings"].get(obj, {})}


def known_objects() -> list[int]:
    return sorted({p["obj"] for p in S["prompts"]} | set(S["masks"]))


def _apply_erase(p: dict) -> int:
    """Clear one erase box from its object's mask; returns pixels removed."""
    m = _unpack(p["obj"], p["frame"])
    if m is None or not m.any():
        return 0
    dw, dh = S["depth_size"]
    rw, rh = S["rgb_size"]
    x1, x2 = sorted((p["x1"], p["x2"]))
    y1, y2 = sorted((p["y1"], p["y2"]))
    px1 = max(int(x1 * dw / rw), 0)
    px2 = min(int(np.ceil(x2 * dw / rw)) + 1, dw)
    py1 = max(int(y1 * dh / rh), 0)
    py2 = min(int(np.ceil(y2 * dh / rh)) + 1, dh)
    region = m[py1:py2, px1:px2]
    n = int(region.sum())
    if n:
        region[:] = False
        S["masks"][p["obj"]][p["frame"]] = _pack(m)
    return n


def _apply_erase_click(p: dict) -> int:
    """Erase the connected mask blob under the click; returns pixels removed."""
    m = _unpack(p["obj"], p["frame"])
    if m is None or not m.any():
        return 0
    dw, dh = S["depth_size"]
    rw, rh = S["rgb_size"]
    px = int(p["x"] * dw / rw)
    py = int(p["y"] * dh / rh)
    if not (0 <= px < dw and 0 <= py < dh) or not m[py, px]:
        return 0
    _, lbl = cv2.connectedComponents(m.astype(np.uint8))
    blob = lbl == lbl[py, px]
    m &= ~blob
    S["masks"][p["obj"]][p["frame"]] = _pack(m)
    return int(blob.sum())


def _sam_refine_erase(p: dict, ckpt: str, cfg: str, offload: bool):
    """SAM2 single-frame carve: like a background (-) click, but only for
    this frame's stored mask. The tracked mask is fed back to SAM2 as a mask
    prompt and the click as a negative point; SAM2 re-segments and the
    refined mask replaces this frame only. Returns net pixels removed, or
    None if SAM2 could not run (caller falls back to blob erase).
    """
    m = _unpack(p["obj"], p["frame"])
    if m is None or not m.any():
        return None
    try:
        get_predictor(ckpt, cfg, offload)
        rw, rh = S["rgb_size"]
        dw, dh = S["depth_size"]
        seed = cv2.resize(m.astype(np.uint8), (rw, rh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        # A lone negative point erases everything -- SAM needs to be told
        # where the object still IS. Anchor a positive point at the mask's
        # deep interior, as far from the click as possible.
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        ys, xs = np.nonzero(m)
        cx, cy = p["x"] * dw / rw, p["y"] * dh / rh
        k = int(np.argmax(dt[ys, xs] * np.hypot(xs - cx, ys - cy)))
        anchor = (xs[k] * rw / dw, ys[k] * rh / dh)
        pred, state = S["predictor"], S["state"]
        before = int(m.sum())
        with sam_run():
            pred.reset_state(state)
            S["state_tracked"] = False
            pred.add_new_mask(state, p["frame"], int(p["obj"]), seed)
            f, ids, logits = pred.add_new_points_or_box(
                inference_state=state, frame_idx=p["frame"],
                obj_id=int(p["obj"]),
                points=np.array([anchor, [p["x"], p["y"]]], np.float32),
                labels=np.array([1, 0], np.int32),
                clear_old_points=False)
            _absorb(f, ids, logits, or_merge=False,
                    only_objs={int(p["obj"])})
        after = _unpack(p["obj"], p["frame"])
        refresh_prompt_state(ckpt, cfg, offload, absorb=False)
        return before - int(0 if after is None else after.sum())
    except Exception as e:
        log(f"SAM2 erase-carve failed ({type(e).__name__}: {e}) -- falling "
            "back to deleting the connected blob.")
        traceback.print_exc()
        return None


def apply_erases(ckpt: str | None = None, cfg: str | None = None,
                 offload: bool = False) -> int:
    """Re-apply every stored erase (after tracking); returns erase count."""
    n = 0
    for p in S["prompts"]:
        if p["kind"] == "erase":
            _apply_erase(p)
            n += 1
        elif p["kind"] == "erase_click":
            done = None
            if ckpt is not None and not S["preview_failed"]:
                done = _sam_refine_erase(p, ckpt, cfg, offload)
            if done is None:
                _apply_erase_click(p)
            n += 1
    return n


def split_overlaps() -> int:
    """Make per-frame object masks mutually exclusive; returns frames changed.

    SAM2 sometimes lets crossing objects claim the same pixels ("merged"
    masks). Contested pixels are assigned to the object whose undisputed
    region on that frame is nearest; an object with no undisputed pixels
    (fully swallowed) falls back to its mask from the previous frame, so
    motion continuity decides.
    """
    if len(S["masks"]) < 2:
        return 0
    changed = 0
    prev: dict[int, np.ndarray] = {}
    for f in range(S["n_frames"]):
        cur: dict[int, np.ndarray] = {}
        for o in sorted(S["masks"]):
            m = _unpack(o, f)
            if m is not None and m.any():
                cur[o] = m
        if len(cur) >= 2:
            claims = np.zeros(next(iter(cur.values())).shape, np.uint8)
            for m in cur.values():
                claims += m
            overlap = claims >= 2
            frame_changed = False
            if overlap.any():
                objs_here = sorted(cur)
                # A mask that is (almost) entirely contested got swallowed by
                # another object's blob -- the SAM2 "merge" case. Its claim is
                # the more specific one, so it keeps its whole mask and the
                # swallowing blob gives those pixels up.
                swallowed = [o for o in objs_here
                             if (cur[o] & overlap).sum()
                             >= 0.95 * cur[o].sum()]
                if len(swallowed) == 1:
                    w = swallowed[0]
                    for o in objs_here:
                        if o != w:
                            cur[o] = cur[o] & ~cur[w]
                            S["masks"][o][f] = _pack(cur[o])
                    frame_changed = True
                    claims = np.zeros_like(claims)
                    for m in cur.values():
                        claims += m
                    overlap = claims >= 2
            if overlap.any():
                objs_here = sorted(cur)
                dists = []
                for o in objs_here:
                    excl = cur[o] & (claims == 1)
                    ref = excl if excl.any() else prev.get(o, cur[o])
                    if not ref.any():
                        ref = cur[o]
                    d = cv2.distanceTransform((~ref).astype(np.uint8),
                                              cv2.DIST_L2, 3)
                    dists.append(np.where(cur[o], d, np.inf))
                winner = np.argmin(np.stack(dists), axis=0)
                for k, o in enumerate(objs_here):
                    nm = cur[o] & (~overlap | (winner == k))
                    S["masks"][o][f] = _pack(nm)
                    cur[o] = nm
                frame_changed = True
            if frame_changed:
                changed += 1
        prev.update(cur)
    return changed


# ----------------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------------


def probe(path: str) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return n, w, h, fps


def probe_bits(path: str) -> int:
    """Source bit depth from the pixel format (yuv420p10le -> 10); 8 if unknown."""
    if not shutil.which("ffprobe"):
        return 8
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15).stdout.strip()
        m = re.search(r"(\d+)(?:le|be)\b", out)
        return int(m.group(1)) if m else 8
    except Exception:
        return 8


def extract_frames(video: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-q:v", "2", "-start_number", "0", str(out_dir / "%05d.jpg")],
            check=True)
    else:
        cap = cv2.VideoCapture(video)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            i += 1
        cap.release()
    return len(list(out_dir.glob("*.jpg")))


def rgb_frame(idx: int) -> np.ndarray:
    img = cv2.imread(str(S["frames_dir"] / f"{idx:05d}.jpg"))
    if img is None:
        raise gr.Error(f"Missing extracted frame {idx}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def depth_frame(idx: int) -> np.ndarray:
    """16-bit gray frame for interactive preview.

    Uses the fast OpenCV seek-decoder, which flattens to 8 bits; values are
    expanded to the 16-bit scale (x257) so all mask/treatment math runs in
    one unit system. Renders use iter_depth16, which decodes true 10-bit.
    """
    cap = S["depth_cap"]
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise gr.Error(f"Cannot read depth frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257


def iter_depth16(start: int, count: int | None):
    """Yield uint16 gray depth frames sequentially at full source precision.

    >8-bit sources decode through an ffmpeg gray16le pipe; 8-bit sources (or
    no ffmpeg) fall back to OpenCV and are expanded x257 like depth_frame.
    """
    dw, dh = S["depth_size"]
    if S["depth_bits"] > 8 and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error"]
        if start > 0:
            # seek to mid-gap before the target frame: robust to fps rounding
            fps = S["depth_fps"] or 24.0
            cmd += ["-ss", f"{(start - 0.5) / fps:.6f}"]
        cmd += ["-i", str(S["depth_path"]), "-f", "rawvideo",
                "-pix_fmt", "gray16le"]
        if count is not None:
            cmd += ["-frames:v", str(count)]
        proc = subprocess.Popen(cmd + ["-"], stdout=subprocess.PIPE)
        try:
            nbytes = dw * dh * 2
            while True:
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                yield np.frombuffer(buf, "<u2").reshape(dh, dw).copy()
        finally:
            proc.stdout.close()
            with contextlib.suppress(Exception):
                proc.kill()
            proc.wait()
    else:
        cap = cv2.VideoCapture(str(S["depth_path"]))
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            done = 0
            while count is None or done < count:
                ok, frame = cap.read()
                if not ok:
                    break
                yield (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                       .astype(np.uint16) * 257)
                done += 1
        finally:
            cap.release()


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    for i in range(1, 1000):
        cand = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not cand.exists():
            return cand
    raise gr.Error(f"Could not find a free filename near {p}")


class VideoSink:
    """ffmpeg pipe (libx264 8-bit / libx265 10-bit) with OpenCV fallback.

    bits > 8 expects uint16 gray frames and encodes 10-bit HEVC (yuv420p10le).
    bits == 8 accepts uint8 frames, or uint16 which are reduced with >> 8.
    """

    def __init__(self, path: str, w: int, h: int, fps: float, crf: int,
                 pix_fmt: str = "gray", bits: int = 8):
        self.proc, self.cvw, self.pix_fmt = None, None, pix_fmt
        self.bits = bits if shutil.which("ffmpeg") else 8
        if shutil.which("ffmpeg"):
            if self.bits > 8:
                in_fmt = "gray16le"
                codec = ["-c:v", "libx265", "-preset", "medium",
                         "-crf", str(crf), "-pix_fmt", "yuv420p10le",
                         "-x265-params", "log-level=error", "-tag:v", "hvc1"]
            else:
                in_fmt = pix_fmt
                codec = ["-c:v", "libx264", "-preset", "medium",
                         "-crf", str(crf), "-pix_fmt", "yuv420p"]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", in_fmt,
                 "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
                 *codec, path],
                stdin=subprocess.PIPE)
        else:
            self.cvw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.bits > 8:
            self.proc.stdin.write(
                np.ascontiguousarray(frame, dtype="<u2").tobytes())
            return
        if frame.dtype == np.uint16:
            frame = (frame >> 8).astype(np.uint8)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        elif self.pix_fmt == "gray":
            self.cvw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        else:
            self.cvw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise gr.Error("ffmpeg encoder failed")
        if self.cvw is not None:
            self.cvw.release()


# ----------------------------------------------------------------------------
# Mask math
# ----------------------------------------------------------------------------


def feather_one(mask: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if feather_px > 0:
        k = 2 * feather_px + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def snap_mask_to_depth(mask: np.ndarray, gray: np.ndarray,
                       grow_px: int, tol: float) -> np.ndarray:
    """Grow the mask into adjacent pixels that share the object's depth range.

    DepthCrafter silhouettes often overhang the RGB object by a few (varying)
    pixels, leaving a bright rim outside the SAM2 mask. Geodesic dilation
    fixes that without blanket dilation: sample the object's own depth values
    (from an eroded core so edge bleed doesn't skew the stats), then grow
    outward -- at most grow_px -- only into pixels whose depth matches.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    h, w = mask.shape
    pad = int(grow_px) + 1
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    g = gray[y0:y1, x0:x1].astype(np.float32)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(m, k3, iterations=2)
    vals = g[(core if core.any() else m).astype(bool)]
    lo, hi = np.percentile(vals, (5.0, 95.0))

    # Distance of each pixel's depth from the object's depth interval.
    d_obj = np.maximum(np.maximum(lo - g, g - hi), 0.0)
    cand = d_obj <= tol
    # Depth maps have soft (antialiased) edges, so the overhang fades toward
    # the background and a fixed tolerance stops halfway across it. Flood a
    # marker watershed from the object core vs the crop border: it hands the
    # fading overhang to the object but stops at depth edges, so occluders
    # (e.g. a wall) stay background even when their gray value is object-like.
    border = np.zeros(m.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg_seed = border & (d_obj > tol) & ~m.astype(bool)
    if bg_seed.any():
        markers = np.zeros(m.shape, np.int32)
        markers[bg_seed] = 1
        markers[(core if core.any() else m).astype(bool)] = 2
        g8 = np.clip(g / 257.0 + 0.5, 0, 255).astype(np.uint8)
        ws = cv2.watershed(cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR), markers)
        cand |= ws == 2
    else:
        # object fills the crop -- fall back to a value-based criterion
        ring = (cv2.dilate(m, k3, iterations=pad) > 0) & ~m.astype(bool)
        bg_vals = g[ring & (d_obj > tol)]
        if bg_vals.size:
            bg = float(np.median(bg_vals))
            cand |= d_obj < np.abs(g - bg)
    cand = cand.astype(np.uint8)

    for _ in range(int(grow_px)):
        nxt = (cv2.dilate(m, k3) & cand) | m
        if np.array_equal(nxt, m):
            break
        m = nxt
    out = mask.copy()
    out[y0:y1, x0:x1] = m.astype(bool)
    return out


def apply_mode(gray: np.ndarray, feathered: np.ndarray, mode: str,
               brightness: float, threshold: float, gain: float,
               inpaint_radius: int) -> np.ndarray:
    """gray is uint16 (16-bit scale); threshold stays in 8-bit slider units."""
    d = gray.astype(np.float32)
    if mode == "brightness":
        target = d * brightness
    elif mode == "compress":
        thr = float(threshold) * 257.0
        target = np.where(d > thr, thr + (d - thr) * gain, d)
    else:  # inpaint (cv2.inpaint is 8-bit only; the fill is synthetic anyway)
        hard = (feathered > 0.5).astype(np.uint8) * 255
        if hard.any():
            g8 = (gray >> 8).astype(np.uint8)
            target = cv2.inpaint(g8, hard, int(inpaint_radius),
                                 cv2.INPAINT_TELEA).astype(np.float32) * 257.0
        else:
            target = d
    out = d * (1.0 - feathered) + target * feathered
    return np.clip(out + 0.5, 0, 65535).astype(np.uint16)


def process_depth(gray: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply each object's treatment in ID order; also return combined matte."""
    matte = None
    orig = gray  # pristine depth frame; snapping must not see earlier objects' edits
    for obj in sorted(S["masks"]):
        m = _unpack(obj, idx)
        if m is None or not m.any():
            continue
        st = settings_for(obj)
        if int(st.get("snap_px", 0)) > 0:
            m = snap_mask_to_depth(m, orig, int(st["snap_px"]),
                                   float(st.get("snap_tol", 12)) * 257.0)
        f = feather_one(m, int(st["dilate_px"]), int(st["feather_px"]))
        gray = apply_mode(gray, f, st["mode"], st["brightness"],
                          st["threshold"], st["gain"], int(st["inpaint_radius"]))
        matte = f if matte is None else np.maximum(matte, f)
    return gray, matte


# ----------------------------------------------------------------------------
# SAM2 glue
# ----------------------------------------------------------------------------


def get_predictor(ckpt: str, cfg: str, offload: bool) -> str:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S["device"] = device
    if S["predictor"] is None:
        if device == "cpu":
            gr.Warning("No CUDA GPU detected -- SAM2 will run on CPU and be "
                       "very slow.")
        if not Path(ckpt).exists():
            raise gr.Error(
                f"SAM2 checkpoint not found at '{ckpt}'. Download "
                "sam2.1_hiera_large.pt (see the docstring) and check the "
                "path in Model settings.")
        log("Loading SAM2 model (one-off, please wait)...")
        S["predictor"] = build_sam2_video_predictor(cfg, ckpt, device=device)
        log(f"SAM2 ready on {device}.")
    if S["state"] is None:
        log("Indexing frames for SAM2...")
        S["state"] = S["predictor"].init_state(
            video_path=str(S["frames_dir"]),
            offload_video_to_cpu=offload,
            offload_state_to_cpu=offload)
    return device


@contextlib.contextmanager
def sam_run():
    import torch
    use_bf16 = (S["device"] == "cuda"
                and torch.cuda.get_device_properties(0).major >= 8)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16 else contextlib.nullcontext())
    with torch.inference_mode(), autocast:
        yield


def prompt_groups() -> dict:
    groups: dict[tuple[int, int], dict] = {}
    for p in S["prompts"]:
        if p["kind"].startswith("erase"):  # mask edits, not SAM2 prompts
            continue
        g = groups.setdefault((p["obj"], p["frame"]),
                              {"points": [], "labels": [], "box": None})
        if p["kind"] == "box":
            g["box"] = [p["x1"], p["y1"], p["x2"], p["y2"]]
        else:
            g["points"].append([p["x"], p["y"]])
            g["labels"].append(p["label"])
    return groups


def _sam_add_group(obj: int, frame: int, g: dict):
    pts = np.array(g["points"], np.float32) if g["points"] else None
    lbl = np.array(g["labels"], np.int32) if g["points"] else None
    box = np.array(g["box"], np.float32) if g["box"] else None
    return S["predictor"].add_new_points_or_box(
        inference_state=S["state"], frame_idx=frame, obj_id=obj,
        points=pts, labels=lbl, box=box)


def _absorb(frame: int, obj_ids, logits, or_merge: bool,
            only_objs: set[int] | None = None) -> None:
    dw, dh = S["depth_size"]
    for i, oid in enumerate(obj_ids):
        oid = int(oid)
        if only_objs is not None and oid not in only_objs:
            continue
        m = (logits[i, 0] > 0.0).cpu().numpy()
        m = cv2.resize(m.astype(np.uint8), (dw, dh),
                       interpolation=cv2.INTER_LINEAR).astype(bool)
        d = S["masks"].setdefault(oid, {})
        if or_merge and frame in d:
            m = m | _unpack(oid, frame)
        d[frame] = _pack(m)


def preview_group(obj: int, frame: int, ckpt: str, cfg: str,
                  offload: bool) -> None:
    """Instant single-frame mask preview for one (object, frame) prompt group."""
    if S["preview_failed"]:
        return
    g = prompt_groups().get((obj, frame))
    if g is None:
        return
    try:
        get_predictor(ckpt, cfg, offload)
        # A state that has already been propagated holds tracked masks on
        # every frame; those suppress a new object's preview through SAM2's
        # non-overlap constraint. Rebuild the prompt-only state first.
        if S["state_tracked"]:
            refresh_prompt_state(ckpt, cfg, offload, absorb=False)
            if S["preview_failed"]:
                return
        with sam_run():
            f, ids, logits = _sam_add_group(obj, frame, g)
            _absorb(f, ids, logits, or_merge=False, only_objs={int(obj)})
    except Exception as e:
        S["preview_failed"] = True
        log(f"Instant preview disabled ({type(e).__name__}: {e}) -- points "
            "are still recorded; 'Track objects' will retry SAM2.")
        traceback.print_exc()


def refresh_prompt_state(ckpt: str, cfg: str, offload: bool,
                         absorb: bool = True) -> None:
    """Re-sync SAM2's prompt state with S['prompts'] after removals."""
    if S["state"] is None or S["predictor"] is None or S["preview_failed"]:
        return
    try:
        S["predictor"].reset_state(S["state"])
        S["state_tracked"] = False
        with sam_run():
            for (obj, frame), g in prompt_groups().items():
                f, ids, logits = _sam_add_group(obj, frame, g)
                if absorb:
                    _absorb(f, ids, logits, or_merge=False,
                            only_objs={int(obj)})
    except Exception as e:
        S["preview_failed"] = True
        log(f"SAM2 state refresh failed ({e}) -- re-run tracking before "
            "rendering.")
        traceback.print_exc()


# ----------------------------------------------------------------------------
# View rendering
# ----------------------------------------------------------------------------


def frame_info_md(idx) -> str:
    if S["frames_dir"] is None:
        return ""
    idx = int(idx)
    fps = S["depth_fps"] or 24.0
    pf = sorted({p["frame"] for p in S["prompts"]})
    pf_txt = ", ".join(str(f) for f in pf) if pf else "none"
    return (f"frame **{idx}** / {S['n_frames'] - 1} &nbsp;|&nbsp; "
            f"{idx / fps:.3f} s @ {fps:.2f} fps &nbsp;|&nbsp; "
            f"prompt frames: {pf_txt}")


def overlay_rgb(idx: int, show_masks: bool, opacity: float,
                outline: bool) -> np.ndarray:
    img = rgb_frame(idx).astype(np.float32)
    rw, rh = S["rgb_size"]
    masks_here = []
    if show_masks:
        for obj in sorted(S["masks"]):
            m = _unpack(obj, idx)
            if m is None or not m.any():
                continue
            m8 = cv2.resize(m.astype(np.uint8), (rw, rh),
                            interpolation=cv2.INTER_NEAREST)
            masks_here.append((obj, m8))
    if not outline:
        for obj, m8 in masks_here:
            mf = m8.astype(np.float32)[..., None] * float(opacity)
            img = img * (1 - mf) + np.array(_color(obj), np.float32) * mf
    img = img.astype(np.uint8)
    if outline:
        for obj, m8 in masks_here:
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, _color(obj), 2)

    for p in S["prompts"]:
        if p["frame"] != idx:
            continue
        col = _color(p["obj"])
        if p["kind"] == "erase_click":
            c = (int(p["x"]), int(p["y"]))
            cv2.drawMarker(img, c, (255, 60, 60), cv2.MARKER_TILTED_CROSS,
                           14, 2)
            cv2.putText(img, f"erase {p['obj']}", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1,
                        cv2.LINE_AA)
        elif p["kind"] == "erase":
            x1, y1, x2, y2 = (int(p["x1"]), int(p["y1"]),
                              int(p["x2"]), int(p["y2"]))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 60, 60), 2)
            cv2.putText(img, f"erase {p['obj']}", (x1 + 4, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1,
                        cv2.LINE_AA)
        elif p["kind"] == "box":
            x1, y1, x2, y2 = (int(p["x1"]), int(p["y1"]),
                              int(p["x2"]), int(p["y2"]))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, str(p["obj"]), (x1 + 4, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        else:
            c = (int(p["x"]), int(p["y"]))
            if p["label"] == 1:
                cv2.circle(img, c, 6, col, -1)
                cv2.circle(img, c, 6, (255, 255, 255), 1)
            else:
                cv2.circle(img, c, 7, col, 2)
                cv2.line(img, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4),
                         (255, 60, 60), 2)
                cv2.line(img, (c[0] - 4, c[1] + 4), (c[0] + 4, c[1] - 4),
                         (255, 60, 60), 2)
            cv2.putText(img, str(p["obj"]), (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    pb = S["pending_box"]
    if pb and pb["frame"] == idx:
        cv2.drawMarker(img, (int(pb["x"]), int(pb["y"])), _color(pb["obj"]),
                       cv2.MARKER_CROSS, 16, 2)
    return img


def render_view(idx, view, show_masks, opacity, outline):
    if S["frames_dir"] is None:
        return None
    idx = max(0, min(int(idx), S["n_frames"] - 1))

    if view == "RGB + prompts/masks":
        return overlay_rgb(idx, bool(show_masks), float(opacity),
                           bool(outline))

    gray = depth_frame(idx)
    after, _ = process_depth(gray.copy(), idx)
    gray8 = (gray >> 8).astype(np.uint8)
    after8 = (after >> 8).astype(np.uint8)
    divider = np.full((gray8.shape[0], 4), 255, np.uint8)
    combo = np.hstack([gray8, divider, after8])
    combo = cv2.cvtColor(combo, cv2.COLOR_GRAY2RGB)
    cv2.putText(combo, "before", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(combo, "after", (gray.shape[1] + 14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
    return combo


def refresh(idx, view, show_masks, opacity, outline):
    return (render_view(idx, view, show_masks, opacity, outline),
            frame_info_md(idx))


def prompts_table():
    rows = []
    for p in S["prompts"]:
        if p["kind"] in ("box", "erase"):
            rows.append([p["obj"], p["frame"],
                         f"{p['x1']:.0f}-{p['x2']:.0f}",
                         f"{p['y1']:.0f}-{p['y2']:.0f}", p["kind"]])
        elif p["kind"] == "erase_click":
            rows.append([p["obj"], p["frame"], f"{p['x']:.0f}",
                         f"{p['y']:.0f}", "erase click"])
        else:
            rows.append([p["obj"], p["frame"], f"{p['x']:.0f}",
                         f"{p['y']:.0f}",
                         "+" if p["label"] == 1 else "-"])
    return rows


def scope_choices() -> list[str]:
    return [DEFAULT_SCOPE] + [f"object {o}" for o in known_objects()]


def scope_dd_update(current: str | None = None):
    ch = scope_choices()
    return gr.update(choices=ch, value=current if current in ch else ch[0])


def _scope_obj(scope) -> int | None:
    if isinstance(scope, str) and scope.startswith("object"):
        return int(scope.split()[-1])
    return None


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------


def blend_ghost_frames(depth_path: str, frames_dir: Path, n: int,
                       ghost: float, progress, lo: float, hi: float) -> None:
    """Blend colormapped depth into the extracted RGB frames (in place).

    Depth is stretched to each frame's size, so lower-res / other-aspect
    depth maps line up the same way masks do (normalized coordinates).
    TURBO turns depth into hue, adding color edges at depth boundaries
    without erasing the natural RGB detail SAM2 was trained on.
    """
    cap = cv2.VideoCapture(depth_path)
    try:
        for i in range(n):
            ok, dfr = cap.read()
            if not ok:
                break
            fp = frames_dir / f"{i:05d}.jpg"
            img = cv2.imread(str(fp))
            if img is None:
                continue
            dg = cv2.resize(cv2.cvtColor(dfr, cv2.COLOR_BGR2GRAY),
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            cm = cv2.applyColorMap(dg, cv2.COLORMAP_TURBO)
            out = cv2.addWeighted(img, 1.0 - ghost, cm, ghost, 0)
            cv2.imwrite(str(fp), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i % 25 == 0:
                progress(lo + (hi - lo) * i / max(n, 1),
                         desc=f"Depth ghost frame {i}/{n}")
    finally:
        cap.release()


def load_videos(rgb_path, depth_path, ghost, view, show_masks, opacity,
                outline, progress=gr.Progress()):
    rgb_path = rgb_path or _cfg_file("rgb")
    depth_path = depth_path or _cfg_file("depth")
    if not rgb_path or not depth_path:
        raise gr.Error("Select both an RGB video and a depth video first.")
    progress(0.05, desc="Probing videos")
    _, rw, rh, _ = probe(rgb_path)
    dn, dw, dh, dfps = probe(depth_path)
    dbits = probe_bits(depth_path)
    if dbits > 8 and not shutil.which("ffmpeg"):
        gr.Warning(f"Depth video is {dbits}-bit but ffmpeg was not found -- "
                   "rendering will fall back to 8-bit. Install ffmpeg to "
                   "preserve depth precision: winget install Gyan.FFmpeg")
    if abs(rw / rh - dw / dh) > 0.02:
        gr.Warning("RGB and depth aspect ratios differ -- masks may not "
                   "line up. Double-check the before|after view.")
        log("Warning: RGB and depth aspect ratios differ.")

    if S["frames_dir"]:
        shutil.rmtree(S["frames_dir"], ignore_errors=True)
    if S["depth_cap"]:
        S["depth_cap"].release()

    progress(0.15, desc="Extracting RGB frames")
    frames_dir = Path(tempfile.mkdtemp(prefix="sam2gui_"))
    rn = extract_frames(rgb_path, frames_dir)
    if rn == 0:
        raise gr.Error("No frames extracted from the RGB video.")
    n = min(rn, dn) if dn > 0 else rn
    if dn > 0 and rn != dn:
        gr.Warning(f"Frame counts differ (RGB {rn}, depth {dn}); "
                   f"using the first {n}.")
        log(f"Warning: frame counts differ (RGB {rn}, depth {dn}); "
            f"using {n}.")

    ghost = float(ghost or 0) / 100.0
    if ghost > 0:
        progress(0.55, desc="Blending depth ghost into RGB frames")
        blend_ghost_frames(depth_path, frames_dir, n, ghost,
                           progress, 0.55, 0.95)

    S.update(frames_dir=frames_dir, n_frames=n, rgb_size=(rw, rh),
             depth_path=depth_path, depth_size=(dw, dh), depth_fps=dfps,
             depth_bits=dbits, ghost=ghost,
             depth_cap=cv2.VideoCapture(depth_path),
             prompts=[], masks={}, obj_settings={}, pending_box=None,
             sel_row=None, state=None, preview_failed=False,
             state_tracked=False, cancel=False)
    save_config(rgb=str(rgb_path), depth=str(depth_path),
                ghost=int(ghost * 100))

    progress(1.0)
    log(f"Loaded {n} frames | RGB {rw}x{rh} | depth {dw}x{dh} "
        f"{dbits}-bit @ {dfps:.2f} fps"
        + (" (full precision kept through render)" if dbits > 8 else "")
        + (f" | depth ghost {int(ghost * 100)}% blended into the frames "
           "SAM2 sees" if ghost > 0 else "")
        + ". Click each object to mask (step 2).")
    return (log_md(),
            gr.update(maximum=n - 1, value=0),
            render_view(0, view, show_masks, opacity, outline),
            frame_info_md(0),
            prompts_table(),
            scope_dd_update())


def on_click(idx, view, show_masks, opacity, outline, obj, add_mode, scope,
             ckpt, cfg, offload, evt: gr.SelectData):
    require_loaded()
    if view != "RGB + prompts/masks":
        gr.Warning("Switch the view to 'RGB + prompts/masks' to add points.")
        return (gr.update(), gr.update(), gr.update(), gr.update(), log_md())
    idx, obj = int(idx), int(obj)
    x, y = float(evt.index[0]), float(evt.index[1])

    if add_mode.startswith("erase (click"):
        dw, dh = S["depth_size"]
        rw, rh = S["rgb_size"]
        px, py = int(x * dw / rw), int(y * dh / rh)
        owners = []
        if 0 <= px < dw and 0 <= py < dh:
            for o in sorted(S["masks"]):
                mo = _unpack(o, idx)
                if mo is not None and mo[py, px]:
                    owners.append(o)
        if not owners:
            gr.Warning("No mask under that click on this frame.")
        else:
            o = obj if obj in owners else owners[0]
            p = {"kind": "erase_click", "obj": o, "frame": idx,
                 "x": x, "y": y}
            S["prompts"].append(p)
            removed = None
            if not S["preview_failed"]:
                removed = _sam_refine_erase(p, ckpt, cfg, offload)
            if removed is None:
                n_px = _apply_erase_click(p)
                log(f"Erased the {n_px} px connected blob of object {o} on "
                    f"frame {idx} (SAM2 unavailable for a finer carve). "
                    "Kept and re-applied after every tracking run.")
            else:
                log(f"SAM2 carved {removed} px off object {o} on frame "
                    f"{idx} around the click. Kept and re-applied after "
                    "every tracking run (delete its row to stop).")
    elif add_mode.startswith(("box", "erase")):
        pb = S["pending_box"]
        if pb and pb["obj"] == obj and pb["frame"] == idx:
            x1, x2 = sorted((pb["x"], x))
            y1, y2 = sorted((pb["y"], y))
            S["pending_box"] = None
            if add_mode.startswith("erase"):
                p = {"kind": "erase", "obj": obj, "frame": idx,
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                S["prompts"].append(p)
                n_px = _apply_erase(p)
                if n_px:
                    log(f"Erased {n_px} mask px of object {obj} on frame "
                        f"{idx}. The box is kept and re-applied after every "
                        "tracking run (delete its row to stop).")
                else:
                    log(f"Erase box stored for object {obj} on frame {idx} "
                        "-- no mask pixels there yet; it applies after "
                        "tracking.")
            else:
                S["prompts"] = [p for p in S["prompts"]
                                if not (p["kind"] == "box" and p["obj"] == obj
                                        and p["frame"] == idx)]
                S["prompts"].append({"kind": "box", "obj": obj, "frame": idx,
                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                log(f"Box for object {obj} on frame {idx}.")
                preview_group(obj, idx, ckpt, cfg, offload)
        else:
            S["pending_box"] = {"obj": obj, "frame": idx, "x": x, "y": y}
            log(("Erase" if add_mode.startswith("erase") else "Box")
                + ": first corner set -- click the opposite corner "
                  "(same frame and object).")
    else:
        label = 1 if add_mode.startswith("object") else 0
        S["prompts"].append({"kind": "point", "obj": obj, "frame": idx,
                             "x": x, "y": y, "label": label})
        preview_group(obj, idx, ckpt, cfg, offload)

    return (*refresh(idx, view, show_masks, opacity, outline),
            prompts_table(), scope_dd_update(scope), log_md())


def _drop_prompt(index: int, ckpt, cfg, offload) -> None:
    p = S["prompts"].pop(index)
    if p["kind"].startswith("erase"):
        log(f"Removed {p['kind'].replace('_', ' ')} (object {p['obj']}, "
            f"frame {p['frame']}) -- the erased pixels come back after the "
            "next tracking run.")
        return
    key = (p["obj"], p["frame"])
    refresh_prompt_state(ckpt, cfg, offload)
    if key not in prompt_groups():
        S["masks"].get(p["obj"], {}).pop(p["frame"], None)
        if not S["masks"].get(p["obj"], True):
            S["masks"].pop(p["obj"], None)
    log(f"Removed {p['kind']} prompt (object {p['obj']}, "
        f"frame {p['frame']}). Re-run tracking to update other frames.")


def undo_point(idx, view, show_masks, opacity, outline, scope,
               ckpt, cfg, offload):
    if S["pending_box"]:
        S["pending_box"] = None
        log("Cancelled pending box corner.")
    elif S["prompts"]:
        _drop_prompt(len(S["prompts"]) - 1, ckpt, cfg, offload)
    else:
        gr.Warning("Nothing to undo.")
    S["sel_row"] = None
    return (*refresh(idx, view, show_masks, opacity, outline),
            prompts_table(), scope_dd_update(scope), log_md())


def delete_selected(idx, view, show_masks, opacity, outline, scope,
                    ckpt, cfg, offload):
    r = S["sel_row"]
    if r is None or not (0 <= r < len(S["prompts"])):
        gr.Warning("Click a row in the prompts table first, then delete.")
    else:
        _drop_prompt(r, ckpt, cfg, offload)
        S["sel_row"] = None
    return (*refresh(idx, view, show_masks, opacity, outline),
            prompts_table(), scope_dd_update(scope), log_md())


def clear_points(idx, view, show_masks, opacity, outline):
    S.update(prompts=[], masks={}, obj_settings={}, pending_box=None,
             sel_row=None, state_tracked=False)
    if S["state"] is not None and S["predictor"] is not None:
        with contextlib.suppress(Exception):
            S["predictor"].reset_state(S["state"])
    log("Cleared all prompts, masks and per-object overrides.")
    return (*refresh(idx, view, show_masks, opacity, outline),
            prompts_table(), scope_dd_update(), log_md())


def on_table_select(evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row is None or not (0 <= row < len(S["prompts"])):
        return gr.update()
    S["sel_row"] = int(row)
    return gr.update(value=S["prompts"][int(row)]["frame"])


def new_object():
    objs = known_objects()
    return gr.update(value=(max(objs) + 1) if objs else 1)


def split_now(idx, view, show_masks, opacity, outline):
    require_loaded()
    if len(S["masks"]) < 2:
        gr.Warning("Need at least two tracked objects to split overlaps.")
        n = 0
    else:
        n = split_overlaps()
    log(f"Split overlapping masks on {n} frame(s)." if n
        else "No overlapping masks found.")
    return (*refresh(idx, view, show_masks, opacity, outline), log_md())


def cancel_op():
    S["cancel"] = True
    log("Cancel requested -- stopping after the current frame...")
    return log_md()


def run_tracking(idx, view, show_masks, opacity, outline,
                 ckpt, cfg, offload, track_obj, progress=gr.Progress()):
    require_loaded()
    sam_prompts = [p for p in S["prompts"]
                   if not p["kind"].startswith("erase")]
    if not sam_prompts:
        raise gr.Error("Click at least one point or box on an object first "
                       "(erase boxes alone don't define objects).")
    only = int(track_obj)
    if only and only not in {p["obj"] for p in sam_prompts}:
        raise gr.Error(f"No prompts exist for object {only}.")
    S["cancel"] = False
    try:
        progress(0.02, desc="Loading SAM2")
        get_predictor(ckpt, cfg, offload)
        S["preview_failed"] = False
        pred, state = S["predictor"], S["state"]
        groups = prompt_groups()
        n = S["n_frames"]

        # SAM2 propagates every object in the state over every frame of a
        # pass, so an object first prompted mid-video would accumulate
        # garbage memory on the frames before its prompt (and die right
        # after it). Partition objects by their first prompt frame and run
        # one clean propagation pass per group, each anchored at its own
        # entry frame.
        first = {}
        for (obj, frame) in groups:
            first[obj] = min(frame, first.get(obj, 1 << 30))
        objs = [only] if only else sorted(first)
        parts: dict[int, list[int]] = {}
        for o in objs:
            parts.setdefault(first[o], []).append(o)

        cancelled = False
        n_parts = len(parts)
        for k, anchor in enumerate(sorted(parts)):
            objs_here = set(parts[anchor])
            pred.reset_state(state)
            S["state_tracked"] = False
            with sam_run():
                for (obj, frame), g in groups.items():
                    if obj in objs_here:
                        _sam_add_group(obj, frame, g)
            for o in objs_here:
                S["masks"].pop(o, None)

            max_prompt = max(f for (o, f) in groups if o in objs_here)
            steps = max((n - anchor) + max_prompt, 1)
            tag = (f"group {k + 1}/{n_parts} " if n_parts > 1 else "")
            done = 0

            def tick(i, direction=""):
                frac = (k + min(1.0, done / steps)) / n_parts
                progress(0.03 + 0.95 * frac,
                         desc=f"Tracking {tag}{direction}frame {i}")

            with sam_run():
                for i, ids, logits in pred.propagate_in_video(state):
                    if S["cancel"]:
                        cancelled = True
                        break
                    _absorb(i, ids, logits, or_merge=True,
                            only_objs=objs_here)
                    done += 1
                    tick(i)
                if max_prompt > 0 and not cancelled:
                    for i, ids, logits in pred.propagate_in_video(
                            state, start_frame_idx=max_prompt, reverse=True):
                        if S["cancel"]:
                            cancelled = True
                            break
                        _absorb(i, ids, logits, or_merge=True,
                                only_objs=objs_here)
                        done += 1
                        tick(i, "(reverse) ")
            S["state_tracked"] = True
            if cancelled:
                break

        fixes_txt = ""
        if not cancelled:
            n_er = apply_erases(ckpt, cfg, offload)
            n_split = split_overlaps()
            if n_er:
                fixes_txt += f" {n_er} erase box(es) re-applied."
            if n_split:
                fixes_txt += (f" Overlapping masks split on {n_split} "
                              "frame(s).")

        covered = len({f for _o, d in S["masks"].items()
                       for f, pk in d.items() if pk.any()})
        scope_txt = f"object {only}" if only else "all objects"
        parts_txt = (f" in {n_parts} passes (objects grouped by first "
                     "prompt frame)" if n_parts > 1 else "")
        if cancelled:
            log(f"Tracking cancelled ({scope_txt}); masks kept on "
                f"{covered}/{n} frames so far.")
        else:
            empty = n - covered
            log(f"Tracking done ({scope_txt}){parts_txt}: masks on "
                f"{covered}/{n} frames"
                + (f" ({empty} empty -- fine if objects leave the frame, "
                   "otherwise add corrective clicks there and track again)."
                   if empty else ".") + fixes_txt)
        return (log_md(),
                *refresh(idx, view, show_masks, opacity, outline))
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Tracking failed:\n" + traceback.format_exc())


# --- depth treatment settings ------------------------------------------------


def on_setting_change(scope, mode_v, brt, thr, gain_v, inpr, dil, fea,
                      snapg, snapt, idx, view, show_masks, opacity, outline):
    vals = {"mode": mode_v, "brightness": float(brt),
            "threshold": float(thr), "gain": float(gain_v),
            "inpaint_radius": int(inpr), "dilate_px": int(dil),
            "feather_px": int(fea), "snap_px": int(snapg),
            "snap_tol": float(snapt)}
    o = _scope_obj(scope)
    if o is None:
        S["settings"] = vals
    else:
        S["obj_settings"][o] = vals
    return refresh(idx, view, show_masks, opacity, outline)


def on_scope(scope):
    o = _scope_obj(scope)
    st = settings_for(o) if o is not None else S["settings"]
    return tuple(gr.update(value=st[k]) for k in SET_KEYS)


def clear_override(scope, idx, view, show_masks, opacity, outline):
    o = _scope_obj(scope)
    if o is None:
        gr.Warning("Pick an object in 'Settings for' to remove its override.")
    elif S["obj_settings"].pop(o, None) is not None:
        log(f"Removed treatment override for object {o} (uses default now).")
    st = settings_for(o) if o is not None else S["settings"]
    return (*[gr.update(value=st[k]) for k in SET_KEYS],
            *refresh(idx, view, show_masks, opacity, outline), log_md())


def depth_histogram(idx):
    require_loaded()
    gray = (depth_frame(int(idx)) >> 8).astype(np.uint8)
    W, H, pad = 512, 200, 26
    img = np.full((H + pad, W, 3), 22, np.uint8)

    def draw(counts, color):
        mx = counts.max() or 1.0
        for b in range(256):
            h = int(counts[b] / mx * (H - 6))
            if h:
                cv2.rectangle(img, (b * 2, H - h), (b * 2 + 1, H), color, -1)

    draw(np.bincount(gray.ravel(), minlength=256).astype(np.float32),
         (110, 110, 110))
    u = union_mask(int(idx))
    if u is not None and u.any():
        draw(np.bincount(gray[u].ravel(), minlength=256).astype(np.float32),
             (255, 200, 0))
    else:
        cv2.putText(img, "no mask on this frame", (150, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
    thr = int(S["settings"]["threshold"])
    cv2.line(img, (thr * 2, 0), (thr * 2, H), (255, 80, 80), 1)
    cv2.putText(img, f"thr {thr}", (min(thr * 2 + 5, W - 70), 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "gray = whole frame   amber = masked pixels "
                     "(each normalized)", (6, H + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                cv2.LINE_AA)
    return img


# --- previews & render --------------------------------------------------------


def overlay_clip(idx, span, opacity, outline, progress=gr.Progress()):
    require_loaded()
    S["cancel"] = False
    idx, span = int(idx), int(span)
    lo, hi = max(0, idx - span), min(S["n_frames"] - 1, idx + span)
    rw, rh = S["rgb_size"]
    path = str(Path(tempfile.gettempdir()) / f"s2dm_overlay_{lo}_{hi}.mp4")
    sink = VideoSink(path, rw, rh, S["depth_fps"], 18, pix_fmt="rgb24")
    try:
        for k, i in enumerate(range(lo, hi + 1)):
            if S["cancel"]:
                break
            sink.write(overlay_rgb(i, True, float(opacity), bool(outline)))
            progress((k + 1) / (hi - lo + 1), desc=f"Overlay frame {i}")
    finally:
        sink.close()
    log(f"Overlay preview clip: frames {lo}-{hi}.")
    return path, log_md()


def test_render(idx, span, crf, progress=gr.Progress()):
    require_loaded()
    if not S["masks"]:
        raise gr.Error("Run tracking first.")
    S["cancel"] = False
    idx, span = int(idx), int(span)
    lo, hi = max(0, idx - span), min(S["n_frames"] - 1, idx + span)
    dw, dh = S["depth_size"]
    path = str(Path(tempfile.gettempdir()) / f"s2dm_test_{lo}_{hi}.mp4")
    # decode at full precision like the real render, but encode the preview
    # clip as plain 8-bit H.264 so it plays inline in any browser
    sink = VideoSink(path, dw, dh, S["depth_fps"], int(crf))
    gen = iter_depth16(lo, hi - lo + 1)
    try:
        for k, gray in enumerate(gen):
            if S["cancel"]:
                break
            i = lo + k
            gray, _ = process_depth(gray, i)
            sink.write(gray)
            progress((k + 1) / (hi - lo + 1), desc=f"Test frame {i}")
    finally:
        gen.close()
        sink.close()
    log(f"Test render: frames {lo}-{hi} with current settings.")
    return path, log_md()


def render_video(out_path, crf, want_matte, progress=gr.Progress()):
    require_loaded()
    if not S["masks"]:
        raise gr.Error("Run tracking first.")
    S["cancel"] = False
    req = Path(str(out_path).strip() or "depth_masked.mp4")
    req.parent.mkdir(parents=True, exist_ok=True)
    out = unique_path(req)
    if out != req:
        log(f"'{req}' already exists -- writing '{out}' instead.")

    dw, dh = S["depth_size"]
    out_bits = 10 if S["depth_bits"] > 8 else 8
    sink = VideoSink(str(out), dw, dh, S["depth_fps"], int(crf),
                     bits=out_bits)
    msink = mpath = None
    if want_matte:
        mpath = unique_path(out.with_name(out.stem + "_matte" + out.suffix))
        msink = VideoSink(str(mpath), dw, dh, S["depth_fps"], int(crf))

    gen = iter_depth16(0, None)
    changed = i = 0
    cancelled = False
    try:
        for gray in gen:
            if S["cancel"]:
                cancelled = True
                break
            matte = None
            if i < S["n_frames"]:
                gray, matte = process_depth(gray, i)
                if matte is not None:
                    changed += 1
            sink.write(gray)
            if msink is not None:
                m8 = ((np.clip(matte, 0, 1) * 255).astype(np.uint8)
                      if matte is not None else np.zeros((dh, dw), np.uint8))
                msink.write(m8)
            i += 1
            if S["n_frames"]:
                progress(min(0.99, i / S["n_frames"]),
                         desc=f"Rendering frame {i}")
    finally:
        gen.close()
        sink.close()
        if msink is not None:
            msink.close()

    save_config(out_path=str(req), settings=S["settings"])
    enc = ("10-bit HEVC" if sink.bits > 8 else "8-bit H.264")
    if cancelled:
        log(f"Render cancelled after {i} frames -- partial file: {out}.")
    else:
        log(f"Rendered {out} ({changed}/{i} frames modified, {enc}, "
            f"CRF {int(crf)})." + (f" Matte: {mpath}." if mpath else ""))
    if out_bits > 8 and sink.bits == 8:
        gr.Warning("Source depth is 10-bit but ffmpeg was not found -- "
                   "output fell back to 8-bit.")
    if not shutil.which("ffmpeg"):
        gr.Warning("Rendered with OpenCV (ffmpeg not found). For the "
                   "cleanest depth encode, install ffmpeg: "
                   "winget install Gyan.FFmpeg")
    return log_md(), str(out)


# --- project save / load -------------------------------------------------------


def _norm_proj_path(path: str) -> Path:
    p = Path(str(path).strip() or "project.npz")
    if p.suffix != ".npz":
        p = p.with_suffix(p.suffix + ".npz")
    return p


def save_project(path):
    require_loaded()
    if not S["prompts"] and not S["masks"]:
        raise gr.Error("Nothing to save yet -- add prompts or track first.")
    p = _norm_proj_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"prompts": S["prompts"], "settings": S["settings"],
            "obj_settings": {str(k): v for k, v in S["obj_settings"].items()},
            "depth_size": list(S["depth_size"]), "n_frames": S["n_frames"],
            "depth_path": str(S["depth_path"])}
    arrays = {"meta": np.array(json.dumps(meta))}
    for obj, d in S["masks"].items():
        frames = sorted(d)
        if not frames:
            continue
        arrays[f"o{obj}_f"] = np.array(frames, np.int64)
        arrays[f"o{obj}_m"] = np.stack([d[f] for f in frames])
    np.savez_compressed(p, **arrays)
    save_config(project=str(p))
    log(f"Project saved to {p} ({len(S['prompts'])} prompts, "
        f"{sum(len(d) for d in S['masks'].values())} frame masks).")
    return log_md()


def load_project(path, idx, view, show_masks, opacity, outline,
                 ckpt, cfg, offload):
    require_loaded()
    p = _norm_proj_path(path)
    if not p.exists():
        raise gr.Error(f"Project file not found: {p}")
    data = np.load(p, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if (tuple(meta["depth_size"]) != tuple(S["depth_size"])
            or meta["n_frames"] != S["n_frames"]):
        raise gr.Error(
            f"Project was saved for a {meta['depth_size'][0]}x"
            f"{meta['depth_size'][1]} depth video with {meta['n_frames']} "
            "frames -- it does not match the currently loaded videos.")

    S["prompts"] = meta["prompts"]
    S["settings"] = {**DEFAULTS, **_migrate_settings(meta.get("settings", {}))}
    S["obj_settings"] = {int(k): _migrate_settings(v)
                         for k, v in meta.get("obj_settings", {}).items()}
    S["masks"] = {}
    S["pending_box"] = None
    S["sel_row"] = None
    S["state_tracked"] = False
    for key in data.files:
        if key.endswith("_f") and key.startswith("o"):
            obj = int(key[1:-2])
            frames, bits = data[key], data[f"o{obj}_m"]
            S["masks"][obj] = {int(f): bits[k].copy()
                               for k, f in enumerate(frames)}
    refresh_prompt_state(ckpt, cfg, offload, absorb=False)
    log(f"Project loaded from {p}: {len(S['prompts'])} prompts, "
        f"{sum(len(d) for d in S['masks'].values())} frame masks.")
    return (log_md(),
            *refresh(idx, view, show_masks, opacity, outline),
            prompts_table(), scope_dd_update(),
            *[gr.update(value=S["settings"][k]) for k in SET_KEYS])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

HEAD_KEYS = """
<script>
document.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable)
    return;
  const hit = (id) => {
    const b = document.getElementById(id);
    if (b) { b.click(); e.preventDefault(); }
  };
  if (e.key === 'ArrowLeft')  hit(e.shiftKey ? 'nav-m10' : 'nav-m1');
  if (e.key === 'ArrowRight') hit(e.shiftKey ? 'nav-p10' : 'nav-p1');
});
</script>
"""


def _nav(delta: int):
    def go(idx):
        if S["frames_dir"] is None:
            return gr.update()
        return gr.update(
            value=int(np.clip(int(idx) + delta, 0, S["n_frames"] - 1)))
    return go


with gr.Blocks(title="SAM2 depth masker") as demo:
    gr.Markdown("## SAM2 depth masker\n"
                "Track objects in the RGB video, then brighten/dim, "
                "compress or inpaint them in the matching depth-map video. "
                "Keyboard: **←/→** step 1 frame, **Shift+←/→** step 10.")

    def _file_label(step: str, key: str) -> str:
        p = _cfg_file(key)
        return step + (f" (empty = reuse {Path(p).name})" if p else "")

    with gr.Row():
        rgb_in = gr.File(label=_file_label("1a. RGB video", "rgb"),
                         type="filepath", file_types=["video"])
        depth_in = gr.File(label=_file_label("1b. Depth video", "depth"),
                           type="filepath", file_types=["video"])
        with gr.Column(scale=0, min_width=220):
            ghost_sl = gr.Slider(
                0, 70, value=int(CFG.get("ghost", 0)), step=5,
                label="depth ghost (%)",
                info="blend colormapped depth into the frames SAM2 sees, "
                     "so depth edges guide the masks (0 = off, try 35)")
            load_btn = gr.Button("Load videos", variant="primary")
    status_log = gr.Markdown("Load the two videos to begin.")

    viewer = gr.Image(label="Viewer (click to add points)", type="numpy",
                      interactive=False, height=600)
    with gr.Row():
        btn_m10 = gr.Button("-10", size="sm", scale=0, elem_id="nav-m10")
        btn_m1 = gr.Button("-1", size="sm", scale=0, elem_id="nav-m1")
        frame_slider = gr.Slider(0, 1, step=1, value=0, label="Frame",
                                 scale=8)
        btn_p1 = gr.Button("+1", size="sm", scale=0, elem_id="nav-p1")
        btn_p10 = gr.Button("+10", size="sm", scale=0, elem_id="nav-p10")
    frame_info = gr.Markdown("")
    with gr.Row():
        view_radio = gr.Radio(
            ["RGB + prompts/masks", "Depth before | after"],
            value="RGB + prompts/masks", label="View", scale=2)
        show_chk = gr.Checkbox(value=True, label="show masks")
        opacity_sl = gr.Slider(0.1, 1.0, value=0.45, step=0.05,
                               label="overlay opacity")
        outline_chk = gr.Checkbox(value=False, label="outline only")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**2. Prompts** -- click each object on any clear "
                        "frame (tracking runs both directions from it). "
                        "Each object gets its own ID and color; background "
                        "(-) clicks fix over-segmentation; box mode = click "
                        "two opposite corners. The mask preview appears "
                        "instantly. 'erase (box)' cuts pixels out of the "
                        "current object's mask on the current frame only; "
                        "'erase (click)' works like a background (-) click "
                        "but for one frame only -- SAM2 carves the clicked "
                        "region out of the tracked mask (any object, no "
                        "re-track needed). Both are saved and re-applied "
                        "after every tracking run. Masks of crossing "
                        "objects are kept mutually exclusive automatically "
                        "after tracking.")
            with gr.Row():
                obj_id = gr.Number(value=1, precision=0, label="Object ID")
                new_obj_btn = gr.Button("New object")
                label_radio = gr.Radio(
                    ["object (+)", "background (-)", "box (2 clicks)",
                     "erase (box, this frame)", "erase (click, this frame)"],
                    value="object (+)", label="Click adds", scale=2)
            with gr.Row():
                undo_btn = gr.Button("Undo last")
                del_btn = gr.Button("Delete selected")
                clear_btn = gr.Button("Clear all")
            table = gr.Dataframe(
                headers=["obj", "frame", "x", "y", "type"],
                label="Prompts (click a row to jump to its frame / select "
                      "it for deletion)",
                interactive=False)
            with gr.Row():
                track_obj = gr.Number(value=0, precision=0,
                                      label="Track object (0 = all)")
                track_btn = gr.Button("3. Track objects (run SAM2)",
                                      variant="primary", scale=2)
                cancel_btn = gr.Button("Cancel", variant="stop")
            split_btn = gr.Button("Split overlapping masks (crossing "
                                  "objects; also runs after tracking)")
            with gr.Accordion("Preview clip", open=False):
                span_sl = gr.Slider(5, 120, value=24, step=1,
                                    label="span: current frame ± N")
                with gr.Row():
                    prev_clip_btn = gr.Button("Preview overlay (RGB)")
                    test_btn = gr.Button("Test render (depth)")
                preview_video = gr.Video(label="Preview clip",
                                         interactive=False)
            with gr.Accordion("Project save / load", open=False):
                proj_path = gr.Textbox(value=CFG.get("project", "project.npz"),
                                       label="Project file (.npz) -- prompts, "
                                             "masks and settings")
                with gr.Row():
                    save_proj_btn = gr.Button("Save project")
                    load_proj_btn = gr.Button("Load project")

        with gr.Column():
            gr.Markdown("**4. Depth treatment** -- switch view to "
                        "'Depth before | after' and tune. Settings can be "
                        "overridden per object. If the depth blob overhangs "
                        "the RGB mask (bright rim survives), raise 'snap to "
                        "depth' -- it grows each mask only into neighboring "
                        "pixels of similar depth, so it hugs the depth "
                        "object's real outline instead of dilating blindly.")
            with gr.Row():
                scope_dd = gr.Dropdown(choices=[DEFAULT_SCOPE],
                                       value=DEFAULT_SCOPE,
                                       label="Settings for", scale=2)
                clear_ovr_btn = gr.Button("Remove override")
            mode = gr.Radio(["brightness", "compress", "inpaint"],
                            value=S["settings"]["mode"], label="Mode")
            brightness = gr.Slider(0.0, 2.0, value=S["settings"]["brightness"],
                                   step=0.05,
                                   label="brightness: factor "
                                         "(<1 dims, >1 brightens)")
            threshold = gr.Slider(0, 255, value=S["settings"]["threshold"],
                                  step=1, label="compress: threshold (8-bit)")
            gain = gr.Slider(0.0, 1.0, value=S["settings"]["gain"], step=0.05,
                             label="compress: gain above threshold")
            inpaint_radius = gr.Slider(1, 15,
                                       value=S["settings"]["inpaint_radius"],
                                       step=1, label="inpaint: radius")
            dilate_px = gr.Slider(0, 20, value=S["settings"]["dilate_px"],
                                  step=1, label="mask dilation (px)")
            feather_px = gr.Slider(0, 30, value=S["settings"]["feather_px"],
                                   step=1, label="mask feather (px)")
            snap_px = gr.Slider(0, 40, value=S["settings"]["snap_px"], step=1,
                                label="snap to depth: max grow (px, 0 = off)")
            snap_tol = gr.Slider(0, 64, value=S["settings"]["snap_tol"],
                                 step=1, label="snap: depth tolerance "
                                               "(8-bit gray levels)")
            with gr.Accordion("Depth histogram", open=False):
                hist_btn = gr.Button("Compute for current frame")
                hist_img = gr.Image(label="depth histogram",
                                    interactive=False)

            gr.Markdown("**5. Render** -- existing files are never "
                        "overwritten (a numeric suffix is added).")
            out_path = gr.Textbox(value=CFG.get("out_path",
                                                "depth_masked.mp4"),
                                  label="Output file")
            crf = gr.Slider(0, 23, value=10, step=1,
                            label="CRF (lower = higher quality)")
            matte_chk = gr.Checkbox(value=False,
                                    label="also export mask matte video "
                                          "(*_matte.mp4, for compositing)")
            render_btn = gr.Button("Render depth video", variant="primary")
            out_video = gr.Video(label="Result", interactive=False)

    with gr.Accordion("Model settings", open=False):
        ckpt = gr.Textbox(value="checkpoints/sam2.1_hiera_large.pt",
                          label="SAM2 checkpoint")
        cfg_in = gr.Textbox(value="configs/sam2.1/sam2.1_hiera_l.yaml",
                            label="SAM2 config")
        offload = gr.Checkbox(value=True,
                              label="Offload video/state to CPU RAM "
                                    "(recommended; saves VRAM)")

    # --- wiring ---------------------------------------------------------------

    VIEW = [frame_slider, view_radio, show_chk, opacity_sl, outline_chk]
    SAM = [ckpt, cfg_in, offload]
    setting_comps = [mode, brightness, threshold, gain, inpaint_radius,
                     dilate_px, feather_px, snap_px,
                     snap_tol]  # order matches SET_KEYS
    PROMPT_OUT = [viewer, frame_info, table, scope_dd, status_log]

    for comp in VIEW:
        comp.change(refresh, VIEW, [viewer, frame_info])
    for btn, delta in ((btn_m10, -10), (btn_m1, -1),
                       (btn_p1, 1), (btn_p10, 10)):
        btn.click(_nav(delta), [frame_slider], [frame_slider])

    load_btn.click(load_videos,
                   [rgb_in, depth_in, ghost_sl, view_radio, show_chk,
                    opacity_sl, outline_chk],
                   [status_log, frame_slider, viewer, frame_info, table,
                    scope_dd])

    viewer.select(on_click,
                  VIEW + [obj_id, label_radio, scope_dd] + SAM, PROMPT_OUT)
    new_obj_btn.click(new_object, None, [obj_id])
    undo_btn.click(undo_point, VIEW + [scope_dd] + SAM, PROMPT_OUT)
    del_btn.click(delete_selected, VIEW + [scope_dd] + SAM, PROMPT_OUT)
    clear_btn.click(clear_points, VIEW, PROMPT_OUT)
    table.select(on_table_select, None, [frame_slider])

    track_btn.click(run_tracking, VIEW + SAM + [track_obj],
                    [status_log, viewer, frame_info])
    cancel_btn.click(cancel_op, None, [status_log])
    split_btn.click(split_now, VIEW, [viewer, frame_info, status_log])

    scope_dd.change(on_scope, [scope_dd], setting_comps)
    clear_ovr_btn.click(clear_override, [scope_dd] + VIEW,
                        setting_comps + [viewer, frame_info, status_log])
    for comp in setting_comps:
        comp.input(on_setting_change, [scope_dd] + setting_comps + VIEW,
                   [viewer, frame_info])

    hist_btn.click(depth_histogram, [frame_slider], [hist_img])
    prev_clip_btn.click(overlay_clip,
                        [frame_slider, span_sl, opacity_sl, outline_chk],
                        [preview_video, status_log])
    test_btn.click(test_render, [frame_slider, span_sl, crf],
                   [preview_video, status_log])
    render_btn.click(render_video, [out_path, crf, matte_chk],
                     [status_log, out_video])

    save_proj_btn.click(save_project, [proj_path], [status_log])
    load_proj_btn.click(load_project, [proj_path] + VIEW + SAM,
                        [status_log, viewer, frame_info, table, scope_dd]
                        + setting_comps)

if __name__ == "__main__":
    # allow Gradio to serve the remembered video files back into the pickers
    _allowed = sorted({str(Path(p).parent)
                       for p in (CFG.get("rgb"), CFG.get("depth"))
                       if p and Path(p).exists()})
    demo.queue().launch(inbrowser=True, head=HEAD_KEYS,
                        allowed_paths=_allowed or None)
