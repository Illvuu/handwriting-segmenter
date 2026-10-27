"""
Build real training pairs from (clean original PDF, answered scan PDF) pairs.

For each samples/clean_NNNNN.pdf + samples/answered_NNNNN.pdf:
  1. Render the scan page at its native pixel size (training image = untouched scan pixels).
  2. Render the clean page, register it onto the scan (SIFT + RANSAC homography on printed content),
     warp the clean render into the scan frame.
  3. Derive a 4-class mask in the scan frame:
        0 background
        1 printed      : dark in warped clean AND at least faintly dark in scan
        2 handwriting  : dark in scan, farther than BAND px from any printed pixel
        3 overlap      : (not derivable from a pair -> never emitted; falls into ignore)
      255 ignore       : dark scan pixels within BAND px of printed (misalignment vs real overlap),
                         scanner streaks (tall thin components), anything we cannot trust
  4. Write data/real/image_NNNNN_pP.png, mask_NNNNN_pP.png and debug overlays in data/real/debug/.

Usage (from the repo root):
    python -m scripts.make_real_pairs [--samples DIR] [--out DIR] [--band 3] [--dark 30]
    python -m scripts.make_real_pairs --manifest data/match/manifest.csv [--out DIR] [--status matched]

Manifest mode (from scripts/match_scans.py): one original page may serve many answered scan pages
(several writers, printer + phone). Rows with status "matched" (or whatever --status lists) are
processed; outputs are named image_<quiz>_<scanfile>_p<n>_<src>.png so quiz, writer bundle, page and
source stay recoverable. The scan is rotated upright first (manifest column rot) so training images
are not sideways.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE = 255

# Colour palette for debug overlays (BGR for OpenCV): printed blue, handwriting red, ignore yellow
PALETTE = {1: (255, 80, 0), 2: (0, 0, 255), IGNORE: (0, 220, 255)}


# ----------------------------------------------------------------------------- rendering

def render_page(pdf: pdfium.PdfDocument, index: int, width_px: int | None = None, dpi: float | None = None) -> np.ndarray:
    """Render a page to RGB uint8. Either match a pixel width or render at dpi."""
    page = pdf[index]
    w_pt, h_pt = page.get_size()
    if width_px is not None:
        scale = width_px / w_pt
    else:
        scale = (dpi or 200) / 72.0
    bitmap = page.render(scale=scale, rev_byteorder=True)
    return np.asarray(bitmap.to_pil().convert("RGB"))


def embedded_image_size(pdf: pdfium.PdfDocument, index: int) -> tuple[int, int] | None:
    """Pixel size of the first embedded image on a page (scans are one image per page)."""
    for obj in pdf[index].get_objects():
        if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            try:
                meta = obj.get_info()  # pypdfium2 >= 4
                return int(meta.width), int(meta.height)
            except Exception:
                try:
                    bmp = obj.get_bitmap(render=False)
                    return bmp.width, bmp.height
                except Exception:
                    return None
    return None


# ----------------------------------------------------------------------------- registration

def register(clean_gray: np.ndarray, scan_gray: np.ndarray):
    """
    Homography mapping clean -> scan, estimated from SIFT matches on printed content.
    Returns (H, n_inliers, median_reprojection_error_px).
    """
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(clean_gray, None)
    k2, d2 = sift.detectAndCompute(scan_gray, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        raise RuntimeError(f"too few keypoints: clean={len(k1) if k1 else 0} scan={len(k2) if k2 else 0}")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        raise RuntimeError(f"too few good matches: {len(good)}")

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999)
    if H is None:
        raise RuntimeError("homography failed")
    inl = inl.ravel().astype(bool)
    if inl.sum() < 4:
        raise RuntimeError(f"degenerate homography: {int(inl.sum())} inliers")
    proj = cv2.perspectiveTransform(src[inl], H)
    err = np.linalg.norm(proj.reshape(-1, 2) - dst[inl].reshape(-1, 2), axis=1)
    return H, int(inl.sum()), float(np.median(err)) if len(err) else float("nan")


def refine_ecc(clean_gray: np.ndarray, scan_gray: np.ndarray, H: np.ndarray, levels: int = 1,
               iters: int = 60) -> np.ndarray:
    """
    Sub-pixel refinement of H (clean -> scan) with ECC on blurred images.

    findTransformECC(template, input, W) finds W such that template(x) ~= input(W x), i.e. W maps
    template coords into input coords. With template = scan and input = clean, W = H^-1. We seed it
    with inv(H) and return inv(W) so the caller keeps the clean -> scan convention.

    levels > 1 runs a coarse-to-fine pyramid (factor 2 per level): the coarse levels have a much
    wider basin of convergence, which phone photos (perspective + uneven lighting) need.
    Falls back to H on failure.
    """
    try:
        w = np.linalg.inv(H)
        w = (w / w[2, 2]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)
        for lvl in range(levels - 1, -1, -1):
            f = 2.0 ** lvl
            if f > 1:
                c = cv2.resize(clean_gray, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
                s = cv2.resize(scan_gray, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
            else:
                c, s = clean_gray, scan_gray
            c = cv2.GaussianBlur(c, (5, 5), 0).astype(np.float32) / 255.0
            s = cv2.GaussianBlur(s, (5, 5), 0).astype(np.float32) / 255.0
            # scale the homography into this level: W_l = S * W * S^-1 with S = diag(1/f, 1/f, 1)
            S = np.diag([1 / f, 1 / f, 1.0]).astype(np.float32)
            wl = S @ w @ np.linalg.inv(S)
            wl = (wl / wl[2, 2]).astype(np.float32)
            _, wl = cv2.findTransformECC(s, c, wl, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            w = np.linalg.inv(S) @ wl @ S
            w = (w / w[2, 2]).astype(np.float32)
        h = np.linalg.inv(w.astype(np.float64))
        return h / h[2, 2]
    except (cv2.error, np.linalg.LinAlgError):
        return H


HIT_MAX_GRAY = 100   # was 160: anti-aliased edges and 50% grey fills (render ~128) are not reliably dark in a scan


def printed_hit_rate(clean_warped_gray: np.ndarray, scan_gray: np.ndarray, dark_thr: int = 30) -> float:
    """Fraction of warped-clean printed pixels that are dark in the scan. ~0.9 = good alignment."""
    printed = clean_warped_gray < HIT_MAX_GRAY   # solid glyphs only: grey fills (chessboards, shaded boxes) need not be dark in the scan
    if not printed.any():
        return float("nan")
    dark = darkness(scan_gray) > dark_thr
    return float((dark & printed).sum() / printed.sum())


# ----------------------------------------------------------------------------- masks

def darkness(gray: np.ndarray, bg_kernel: int = 51) -> np.ndarray:
    """How much darker than the local paper background each pixel is (0..255)."""
    bg = cv2.medianBlur(gray, bg_kernel)
    return cv2.subtract(bg, gray)


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def streak_columns(ink: np.ndarray, col_frac: float = 0.12, run_px: int = 150, pad: int = 3,
                   margin_from_x: int | None = None, margin_col_frac: float = 0.03) -> np.ndarray:
    """
    Scanner streaks are perfectly vertical, span most of the page and may be dashed.
    Two detectors, OR-ed, then padded horizontally:
      1. columns whose non-printed ink coverage exceeds col_frac of the page height
      2. any vertical ink run longer than run_px (morphological opening with a 1 x run_px kernel)
    """
    h, w = ink.shape
    u8 = ink.astype(np.uint8)
    profile = u8.sum(axis=0) / h
    cols = profile > col_frac
    if margin_from_x is not None:  # right of all printed content: scanner edge artefacts dominate
        cols[margin_from_x:] |= profile[margin_from_x:] > margin_col_frac
    col_mask = np.zeros_like(ink)
    col_mask[:, cols] = True

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, run_px))
    runs = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k).astype(bool)

    streak = col_mask | runs
    if pad > 0:
        kp = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 1))
        streak = cv2.dilate(streak.astype(np.uint8), kp).astype(bool)
    return streak


PRINTED_MAX_GRAY = 235  # clean render is a digital page on pure white: anything non-white is printed


def build_mask(clean_warped_gray: np.ndarray, scan_gray: np.ndarray, band: int, dark_thr: int,
               faint_thr: int = 15, min_area: int = 4) -> np.ndarray:
    h, w = scan_gray.shape
    # NOT the synthetic `< 160` rule: light-grey grid lines, graph images and tinted boxes are
    # printed too, and if missed they get labelled as handwriting wherever the scan shows them.
    printed_core = clean_warped_gray < PRINTED_MAX_GRAY
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    printed_dilated = cv2.dilate(printed_core.astype(np.uint8), k).astype(bool)
    printed_band = printed_dilated & ~printed_core

    d = darkness(scan_gray)
    scan_dark = d > dark_thr
    scan_dark = remove_small(scan_dark, min_area)
    scan_faint = d > faint_thr

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[printed_core & scan_faint] = 1
    mask[scan_dark & ~printed_dilated] = 2
    mask[printed_band & scan_dark] = IGNORE

    # Large grey fills (chessboards, shaded boxes): the local-background darkness is ~0 inside them,
    # so the fill itself is invisible to `d` and ink on it would be labelled printed by the rule
    # above. Label the fill as printed on an absolute grey test, ink on it as handwriting when it is
    # clearly darker than the fill, and the in-between (halftone texture, faint ink) as ignore.
    # Only regions wider than ~1% of the page count as fills: thin light-grey or coloured printed
    # lines (graph curves, light text) must stay on the normal printed path above, otherwise the
    # IGNORE rule below strips their label and the model learns to erase them (finetuned_d bug).
    printed_light = (printed_core & (clean_warped_gray >= 100)).astype(np.uint8)
    r_fill = max(4, w // 200)
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_fill + 1, 2 * r_fill + 1))
    printed_fill = cv2.morphologyEx(printed_light, cv2.MORPH_OPEN, k_fill).astype(bool)
    fill_inner = cv2.erode(printed_fill.astype(np.uint8), k).astype(bool)   # away from the fill edge
    fill_ink = fill_inner & remove_small(d > 2 * dark_thr, min_area)
    mask[printed_fill & (scan_gray < 225)] = 1
    mask[printed_fill & scan_dark & ~fill_ink] = IGNORE   # fill edges, halftone texture, faint ink
    mask[fill_ink] = 2

    printed_cols = np.flatnonzero(printed_core.any(axis=0))
    margin_x = int(printed_cols.max()) + 20 if printed_cols.size else None
    streaks = streak_columns(scan_dark & ~printed_dilated, margin_from_x=margin_x)
    mask[streaks & (mask != 1)] = IGNORE

    # Outside the clean page content area the scan may show the scanner lid / page edge: anything
    # dark in the outer 1% border that is not printed is untrustworthy.
    bw, bh = max(4, w // 100), max(4, h // 100)
    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = border[-bh:, :] = border[:, :bw] = border[:, -bw:] = True
    mask[border & (mask == 2)] = IGNORE
    return mask


# ----------------------------------------------------------------------------- debug output

def overlay(scan_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    bgr = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2BGR)
    out = bgr.copy()
    for cls, colour in PALETTE.items():
        sel = mask == cls
        out[sel] = (alpha * np.array(colour) + (1 - alpha) * bgr[sel]).astype(np.uint8)
    return out


def alignment_view(scan_gray: np.ndarray, clean_warped_gray: np.ndarray) -> np.ndarray:
    """Scan in grey, warped clean printed pixels in green: misalignment shows as green/grey ghosting."""
    out = cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR)
    printed = clean_warped_gray < PRINTED_MAX_GRAY
    out[printed] = (0, 200, 0)
    return out


def save(path: str, bgr_or_gray: np.ndarray) -> None:
    cv2.imwrite(path, bgr_or_gray)


# ----------------------------------------------------------------------------- main

def rotate90(img: np.ndarray, rot: int) -> np.ndarray:
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def render_scan_page(scan: pdfium.PdfDocument, p: int, rot: int = 0) -> np.ndarray:
    """Scan page at its native pixel size (training image = untouched scan pixels), rotated upright."""
    size = embedded_image_size(scan, p)
    rgb = render_page(scan, p, width_px=size[0] if size else None, dpi=None if size else 200)
    return rotate90(rgb, rot)


def render_clean_like(clean: pdfium.PdfDocument, p: int, sw: int, sh: int) -> np.ndarray:
    rgb = render_page(clean, p, width_px=sw)
    if rgb.shape[1] != sw:  # pdfium rounds the bitmap width independently
        rgb = cv2.resize(rgb, (sw, rgb.shape[0]), interpolation=cv2.INTER_AREA)
    if rgb.shape[0] != sh:  # aspect mismatch: pad/crop clean to scan height
        fixed = np.full((sh, sw, 3), 255, dtype=np.uint8)
        hh = min(sh, rgb.shape[0])
        fixed[:hh] = rgb[:hh]
        rgb = fixed
    return rgb


def process_page(scan_rgb: np.ndarray, clean_rgb: np.ndarray, stem: str, out_dir: str, dbg_dir: str,
                 band: int, dark_thr: int, use_ecc: bool, ecc_levels: int = 1,
                 min_hit: float | None = None) -> dict | None:
    """
    Register clean onto scan, build mask, write image/mask/debug. Returns a stats row or None.
    With min_hit, a page whose printed hit rate stays below it (after ECC) is not written; the row
    comes back with skipped=True and only the alignment view is saved for inspection.
    """
    sh, sw = scan_rgb.shape[:2]
    scan_gray = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2GRAY)
    clean_gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY)

    try:
        H, inliers, med_err = register(clean_gray, scan_gray)
    except RuntimeError as e:
        print(f"  {stem}: REGISTRATION FAILED ({e}); skipping page")
        return None

    def warp(h):
        return cv2.warpPerspective(clean_gray, h, (sw, sh), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    clean_warped = warp(H)
    hit = printed_hit_rate(clean_warped, scan_gray, dark_thr)
    ecc_used = False
    if use_ecc:
        for lv in sorted({1, ecc_levels}):  # plain ECC first, then the pyramid if asked for
            H2 = refine_ecc(clean_gray, scan_gray, H, levels=lv)
            w2 = warp(H2)
            hit2 = printed_hit_rate(w2, scan_gray, dark_thr)
            if hit2 > hit:            # keep ECC only if it measurably helps
                H, clean_warped, hit, ecc_used = H2, w2, hit2, True
    if min_hit is not None and hit < min_hit:
        save(os.path.join(dbg_dir, f"{stem}_align_SKIPPED.png"), alignment_view(scan_gray, clean_warped))
        print(f"  {stem}: SKIPPED printed hit rate {hit:.3f} < {min_hit} after ECC (inliers={inliers})")
        return dict(page=stem, size=f"{sw}x{sh}", inliers=inliers, med_err=med_err, hit=hit, ecc=ecc_used, skipped=True)
    if hit < 0.7:
        print(f"  {stem}: WARNING printed hit rate {hit:.2f} < 0.7, alignment suspect")
    mask = build_mask(clean_warped, scan_gray, band=band, dark_thr=dark_thr)

    Image.fromarray(scan_rgb).save(os.path.join(out_dir, f"image_{stem}.png"))
    Image.fromarray(mask, mode="L").save(os.path.join(out_dir, f"mask_{stem}.png"))
    save(os.path.join(dbg_dir, f"{stem}_overlay.png"), overlay(scan_rgb, mask))
    save(os.path.join(dbg_dir, f"{stem}_align.png"), alignment_view(scan_gray, clean_warped))

    n = mask.size
    pct = {c: 100.0 * np.count_nonzero(mask == c) / n for c in (0, 1, 2, IGNORE)}
    row = dict(page=stem, size=f"{sw}x{sh}", inliers=inliers, med_err=med_err, hit=hit, ecc=ecc_used, skipped=False,
               **{f"c{c}": v for c, v in pct.items()})
    print(f"  {stem}: {sw}x{sh}  inliers={inliers}  med_reproj={med_err:.2f}px  printed_hit={hit:.3f}{' (ecc)' if ecc_used else ''}  "
          f"bg={pct[0]:.1f}%  printed={pct[1]:.2f}%  hw={pct[2]:.2f}%  ignore={pct[IGNORE]:.2f}%")
    return row


def process_pair(clean_pdf: str, answered_pdf: str, pair_id: str, out_dir: str, dbg_dir: str,
                 band: int, dark_thr: int, use_ecc: bool) -> list[dict]:
    clean = pdfium.PdfDocument(clean_pdf)
    scan = pdfium.PdfDocument(answered_pdf)
    if len(clean) != len(scan):
        print(f"  WARNING: page count differs clean={len(clean)} scan={len(scan)}; using min")
    rows = []
    for p in range(min(len(clean), len(scan))):
        scan_rgb = render_scan_page(scan, p)
        sh, sw = scan_rgb.shape[:2]
        clean_rgb = render_clean_like(clean, p, sw, sh)
        row = process_page(scan_rgb, clean_rgb, f"{pair_id}_p{p + 1}", out_dir, dbg_dir, band, dark_thr, use_ecc)
        if row:
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- manifest mode

def slug(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    s = s.replace(" (1)", "").replace(" copy", "")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_")


def read_manifest(path: str) -> list[dict]:
    if path.endswith(".json"):
        with open(path) as fh:
            rows = json.load(fh)
    else:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    for r in rows:
        r["scan_page"] = int(r["scan_page"])
        r["original_page"] = int(r["original_page"] or 0)
        r["rot"] = int(r.get("rot") or 0)
        r["inliers"] = int(r.get("inliers") or 0)
        try:
            r["hit_rate"] = float(r.get("hit_rate") if r.get("hit_rate") not in (None, "") else "nan")
        except (TypeError, ValueError):
            r["hit_rate"] = float("nan")
    return rows


SOURCE_TAG = {"printer": "pr", "phone": "ph"}


def process_manifest(rows: list[dict], statuses: set[str], out_dir: str, dbg_dir: str, band: int,
                     dark_thr: int, dark_phone: int | None, use_ecc: bool, ecc_levels_phone: int,
                     min_hit: float) -> list[dict]:
    rows = [r for r in rows if r.get("status") in statuses and r.get("original_file")]
    docs: dict[str, pdfium.PdfDocument] = {}

    def doc(path):
        if path not in docs:
            docs[path] = pdfium.PdfDocument(path)
        return docs[path]

    out = []
    for r in rows:
        src = r["source"]
        stem = f"{slug(r['original_file'])}_{slug(r['scan_file'])}_p{r['scan_page']}_{SOURCE_TAG.get(src, src)}"
        scan_rgb = render_scan_page(doc(r["scan_file"]), r["scan_page"] - 1, r["rot"])
        sh, sw = scan_rgb.shape[:2]
        clean_rgb = render_clean_like(doc(r["original_file"]), r["original_page"] - 1, sw, sh)
        thr = dark_phone if (src == "phone" and dark_phone is not None) else dark_thr
        lv = ecc_levels_phone if src == "phone" else 1
        row = process_page(scan_rgb, clean_rgb, stem, out_dir, dbg_dir, band, thr, use_ecc, ecc_levels=lv,
                           min_hit=min_hit)
        if row:
            row.update(source=src, original=slug(r["original_file"]), original_page=r["original_page"],
                       scan=slug(r["scan_file"]), scan_page=r["scan_page"])
            out.append(row)
    return out


def print_summary(rows: list[dict]) -> None:
    skipped = [r for r in rows if r.get("skipped")]
    rows = [r for r in rows if not r.get("skipped")]
    if skipped:
        print("\nskipped (hit rate below --min-hit at full resolution; see debug/*_align_SKIPPED.png):")
        for r in skipped:
            print(f"  {r['page']:70s} inliers={r['inliers']} hit={r['hit']:.3f}")
    if not rows:
        return
    def fmt(group):
        n = len(group)
        return (f"n={n:3d}  hit mean={np.mean([g['hit'] for g in group]):.3f} min={min(g['hit'] for g in group):.3f}  "
                f"printed={np.mean([g['c1'] for g in group]):.2f}%  hw={np.mean([g['c2'] for g in group]):.2f}%  "
                f"ignore={np.mean([g['c255'] for g in group]):.2f}%")
    print("\nsummary by source")
    for src in sorted({r["source"] for r in rows}):
        print(f"  {src:8s} {fmt([r for r in rows if r['source'] == src])}")
    print("summary by original (all sources)")
    for o in sorted({r["original"] for r in rows}):
        g = [r for r in rows if r["original"] == o]
        pages = sorted({r["original_page"] for r in g})
        print(f"  {o:40s} {fmt(g)}  orig pages covered={pages}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=os.path.join(BASE_DIR, "samples"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "real"))
    ap.add_argument("--band", type=int, default=3, help="ignore band (px) around printed pixels")
    ap.add_argument("--dark", type=int, default=30, help="darkness-below-background threshold for ink")
    ap.add_argument("--no-ecc", action="store_true", help="skip ECC sub-pixel refinement")
    ap.add_argument("--manifest", default=None, help="manifest.csv/json from scripts.match_scans (see module doc)")
    ap.add_argument("--status", default="matched", help="comma list of manifest statuses to build (manifest mode)")
    ap.add_argument("--dark-phone", type=int, default=None, help="darkness threshold for phone scans (default: --dark)")
    ap.add_argument("--ecc-levels-phone", type=int, default=3, help="ECC pyramid levels tried for phone scans")
    ap.add_argument("--min-hit", type=float, default=0.7, help="manifest mode: skip pages below this printed hit rate")
    args = ap.parse_args()

    dbg_dir = os.path.join(args.out, "debug")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(dbg_dir, exist_ok=True)

    if args.manifest:
        rows = process_manifest(read_manifest(args.manifest), set(args.status.split(",")), args.out, dbg_dir,
                                args.band, args.dark, args.dark_phone, not args.no_ecc, args.ecc_levels_phone,
                                args.min_hit)
        print_summary(rows)
        with open(os.path.join(args.out, "pairs_stats.json"), "w") as fh:
            json.dump(rows, fh, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
        print(f"\nwrote {sum(not r.get('skipped') for r in rows)} pages to {args.out}  (debug overlays in {dbg_dir})")
        return 0

    cleans = sorted(glob.glob(os.path.join(args.samples, "clean_*.pdf")))
    if not cleans:
        print(f"no samples/clean_*.pdf under {args.samples}", file=sys.stderr)
        return 2

    total = 0
    for clean_pdf in cleans:
        m = re.search(r"clean_(\w+)\.pdf$", os.path.basename(clean_pdf))
        pair_id = m.group(1)
        answered_pdf = os.path.join(args.samples, f"answered_{pair_id}.pdf")
        if not os.path.exists(answered_pdf):
            print(f"skip {pair_id}: no {answered_pdf}")
            continue
        print(f"pair {pair_id}")
        rows = process_pair(clean_pdf, answered_pdf, pair_id, args.out, dbg_dir, args.band, args.dark, not args.no_ecc)
        total += len(rows)

    print(f"\nwrote {total} pages to {args.out}  (debug overlays in {dbg_dir})")
    print("check *_align.png: green = printed from clean, should sit exactly on the grey scan glyphs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
