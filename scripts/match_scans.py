"""
Match every page of a bundle of answered scans to the page of the digital original it was printed
from, so one original PDF can serve many answered copies (several writers, several scanners).

Two stages per scan page:
  1. Coarse: SIFT on a small render (~1000 px wide, 2500 features) against every original page,
     ranked by RANSAC inlier count. Cheap enough for 100+ scan pages x 70+ original pages.
  2. Fine: the top-K candidates are re-registered with make_real_pairs.register() at the working
     resolution and scored by the printed hit rate (fraction of printed pixels that are dark in the
     scan after warping). Best = highest hit rate, ties broken by inliers.

A page is "matched" when inliers >= --min-inliers and hit rate >= --min-hit. Everything else is
reported (blank pages, unmatched pages with a thumbnail so the missing original can be identified).

Outputs (default --out data/match):
  manifest.csv / manifest.json   one row per scan page:
      scan_file, scan_page, original_file, original_page, inliers, hit_rate, source, status, rot,
      second_file, second_page, second_hit     (runner-up, to judge ambiguity)
  thumbs/<scan>_p<n>.png         thumbnails of every unmatched / blank scan page
  summary.txt                    per-original coverage and the list of unmatched pages

Usage (from the repo root):
    python -m scripts.match_scans --originals DIR --printer DIR --phone FILE [--out data/match]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pypdfium2 as pdfium

from scripts.make_real_pairs import (darkness, embedded_image_size, printed_hit_rate, register,
                                     render_page)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- helpers

def slug(path: str) -> str:
    """'COMC2022-Problem-Set (1).pdf' -> 'COMC2022-Problem-Set'; 'Scan 1.pdf' -> 'Scan1'."""
    s = os.path.splitext(os.path.basename(path))[0]
    s = s.replace(" (1)", "").replace(" copy", "")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_")


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_width(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if w == width:
        return gray
    return cv2.resize(gray, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def rotation_deg(H: np.ndarray) -> float:
    """Approximate in-plane rotation (deg) of a homography, from its affine part."""
    a, b = H[0, 0], H[1, 0]
    return math.degrees(math.atan2(b, a))


def snap_rot(deg: float) -> int:
    """Nearest multiple of 90 in {0, 90, 180, 270}: how much to rotate the scan to make it upright."""
    r = int(round(deg / 90.0)) * 90 % 360
    return (360 - r) % 360


def rotate90(img: np.ndarray, rot: int) -> np.ndarray:
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def ink_fraction(gray: np.ndarray, dark_thr: int = 30) -> float:
    """Dark-pixel fraction, ignoring scanner streak columns (columns > 30% dark on a blank sheet)."""
    ink = darkness(gray) > dark_thr
    streak = ink.mean(axis=0) > 0.3
    ink[:, streak] = False
    return float(ink.mean())


# ----------------------------------------------------------------------------- feature stage

class Features:
    def __init__(self, nfeatures: int):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def compute(self, gray: np.ndarray):
        k, d = self.sift.detectAndCompute(gray, None)
        return k, d

    def inliers(self, kd_a, kd_b) -> int:
        """RANSAC inlier count for a -> b (0 if not enough matches)."""
        ka, da = kd_a
        kb, db = kd_b
        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            return 0
        pairs = self.matcher.knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return 0
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=2000, confidence=0.995)
        if H is None:
            return 0
        return int(inl.sum())


# ----------------------------------------------------------------------------- fine scoring

def score_candidate(o: dict, gray_fine: np.ndarray, scan_dark: np.ndarray) -> dict | None:
    """
    Register original page o onto the scan at working resolution and score it.
      hit        fraction of printed pixels (clean < 160) that are dark in the scan
      explained  fraction of the scan's dark pixels within 2 px of a printed pixel
      combined   hit + explained: a wrong look-alike page explains only the shared template,
                 the right page explains template + question text + figures
    """
    try:
        H, inl, _ = register(o["fine"], gray_fine)
    except (RuntimeError, cv2.error):
        return None
    h, w = gray_fine.shape
    warped = cv2.warpPerspective(o["fine"], H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    hit = printed_hit_rate(warped, gray_fine)
    hit = 0.0 if math.isnan(hit) else hit
    printed = (warped < 100).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    printed_dil = cv2.dilate(printed, k).astype(bool)
    n_dark = int(scan_dark.sum())
    explained = float((scan_dark & printed_dil).sum() / n_dark) if n_dark else 0.0
    return dict(hit=hit, explained=explained, combined=hit + explained, inl=inl, orig=o,
                rot=snap_rot(rotation_deg(H)), warped=warped, H=H)


_DOCS: dict[str, pdfium.PdfDocument] = {}


def _doc(path: str) -> pdfium.PdfDocument:
    if path not in _DOCS:
        _DOCS[path] = pdfium.PdfDocument(path)
    return _DOCS[path]


def discriminate(close: list[dict], gray_full: np.ndarray, fine_w: int, max_boxes: int = 24) -> None:
    """
    Look-alike pages (continuation sheets, or the same sheet from two competition years) differ only
    in a few glyphs: 'C2' vs 'C3', 'p. 13/20' vs 'p. 15/20', '2021' vs '2022'. At matching
    resolution that is noise, so for the close candidates:
      1. re-render each original at full scan resolution and warp it with the scaled homography,
      2. find the regions where the candidates' solid printed pixels differ (XOR, dilated, boxed),
      3. in each box, correlate each candidate's glyph image against the scan darkness with a small
         shift search (TM_CCOEFF_NORMED), so 1-2 px misregistration does not matter; a candidate
         that prints nothing in a box scores by how much paper the scan shows there.
    Sets c['dscore'] (mean correlation over boxes, higher = better) and c['dboxes'].
    """
    h, w = gray_full.shape
    scale = w / fine_w
    S = np.diag([scale, scale, 1.0])
    d_full = darkness(gray_full).astype(np.float32)
    solids, inks = [], []
    for c in close:
        o = c["orig"]
        clean = to_gray(render_page(_doc(o["file"]), o["page"] - 1, width_px=w))
        if clean.shape[1] != w:
            clean = cv2.resize(clean, (w, clean.shape[0]), interpolation=cv2.INTER_AREA)
        H_full = S @ c["H"] @ np.linalg.inv(S)
        warped = cv2.warpPerspective(clean, H_full, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        solids.append(warped < 100)
        inks.append((255 - warped).astype(np.float32))
    diff = np.zeros_like(solids[0])
    for i in range(len(solids)):
        for j in range(i + 1, len(solids)):
            diff |= solids[i] ^ solids[j]
    diff = cv2.dilate(diff.astype(np.uint8), np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    boxes = [tuple(stats[i, :4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 60]
    boxes.sort(key=lambda b: -b[2] * b[3])
    boxes = boxes[:max_boxes]
    pad, search = 3, 5
    for c, ink in zip(close, inks):
        scores = []
        for (x, y, bw, bh) in boxes:
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            tmpl = ink[y0:y1, x0:x1]
            win = d_full[max(0, y0 - search):min(h, y1 + search), max(0, x0 - search):min(w, x1 + search)]
            if tmpl.std() < 8:  # candidate prints nothing here: right iff the scan shows paper
                scores.append(1.0 - 2.0 * float((d_full[y0:y1, x0:x1] > 30).mean()))
                continue
            if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
                continue
            r = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
            scores.append(float(r.max()))
        c["dboxes"] = len(scores)
        c["dscore"] = float(np.mean(scores)) if scores else 0.0


def match_page(gray_full: np.ndarray, orig: list[dict], feats: Features, top_k: int, margin: float, fine_w: int):
    """Returns (best, runner_up, ambiguous) for one scan page (any orientation); best may be None."""
    gray_fine = resize_width(gray_full, fine_w)
    scan_dark = darkness(gray_fine) > 30
    kd = feats.compute(resize_width(gray_fine, COARSE_W))
    scores = [(feats.inliers(o["kd"], kd), i) for i, o in enumerate(orig)]
    scores.sort(reverse=True)
    cands = []
    for coarse_inl, i in scores[:top_k]:
        if coarse_inl < 10:
            continue
        c = score_candidate(orig[i], gray_fine, scan_dark)
        if c:
            cands.append(c)
    if not cands:
        return None, None, False
    cands.sort(key=lambda c: (c["combined"], c["inl"]), reverse=True)
    best = cands[0]
    close = [c for c in cands if c["combined"] >= best["combined"] - margin]
    if len(close) > 1:
        discriminate(close, gray_full, fine_w)
        close.sort(key=lambda c: (c["dscore"], c["combined"]), reverse=True)
        best, second = close[0], close[1]
        ambiguous = best["dboxes"] == 0 or second["dscore"] >= best["dscore"] - 0.1
        return best, second, ambiguous
    return best, (cands[1] if len(cands) > 1 else None), False


COARSE_W = 1000


# ----------------------------------------------------------------------------- main

def load_originals(paths: list[str], coarse_w: int, fine_w: int, feats: Features):
    """Returns list of dicts, one per original page, with coarse features and fine-res gray."""
    out = []
    for path in paths:
        pdf = pdfium.PdfDocument(path)
        for p in range(len(pdf)):
            rgb = render_page(pdf, p, width_px=fine_w)
            fine = to_gray(rgb)
            coarse = resize_width(fine, coarse_w)
            out.append(dict(file=path, page=p + 1, fine=fine, kd=feats.compute(coarse),
                            ink=ink_fraction(fine)))
        print(f"  original {os.path.basename(path)}: {len(pdf)} pages")
    return out


def scan_pages(printer_dir: str | None, phone_files: list[str]):
    """Yield (source, file, page_index, pdf) for every scan page."""
    items = []
    if printer_dir:
        for f in sorted(glob.glob(os.path.join(printer_dir, "*.pdf"))):
            items.append(("printer", f))
    for f in phone_files:
        items.append(("phone", f))
    for src, f in items:
        pdf = pdfium.PdfDocument(f)
        for p in range(len(pdf)):
            yield src, f, p, pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--originals", required=True, help="directory of digital original PDFs")
    ap.add_argument("--printer", default=None, help="directory of flatbed/ADF scan PDFs")
    ap.add_argument("--phone", nargs="*", default=[], help="phone-camera scan PDF(s)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "match"))
    ap.add_argument("--coarse-width", type=int, default=1000)
    ap.add_argument("--fine-width", type=int, default=1275, help="working width for registration + hit rate")
    ap.add_argument("--coarse-features", type=int, default=2500)
    ap.add_argument("--top-k", type=int, default=12, help="candidates re-scored at fine resolution")
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--min-hit", type=float, default=0.7)
    ap.add_argument("--ambiguous-margin", type=float, default=0.15,
                    help="candidates within this of the best combined score are re-ranked by unique-pixel hit")
    ap.add_argument("--only", default=None, help="comma list of <scanslug>_p<n> to process (debugging)")
    ap.add_argument("--blank-ink", type=float, default=0.002, help="ink fraction below which a scan page is blank")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    thumb_dir = os.path.join(args.out, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    global COARSE_W
    COARSE_W = args.coarse_width
    feats = Features(args.coarse_features)
    originals = sorted(glob.glob(os.path.join(args.originals, "*.pdf")))
    if not originals:
        print(f"no PDFs under {args.originals}", file=sys.stderr)
        return 2
    print(f"loading {len(originals)} originals")
    t0 = time.time()
    orig = load_originals(originals, args.coarse_width, args.fine_width, feats)
    print(f"  {len(orig)} original pages, {time.time() - t0:.0f}s")

    rows = []
    only = set(args.only.split(",")) if args.only else None
    for src, f, p, pdf in scan_pages(args.printer, args.phone):
        if only and f"{slug(f)}_p{p + 1}" not in only:
            continue
        t = time.time()
        size = embedded_image_size(pdf, p)
        rgb = render_page(pdf, p, width_px=size[0] if size else None, dpi=None if size else 200)
        gray_full = to_gray(rgb)
        gray_fine = resize_width(gray_full, args.fine_width)
        row = dict(scan_file=f, scan_page=p + 1, source=src, original_file="", original_page=0,
                   inliers=0, hit_rate=float("nan"), explained=float("nan"), dscore=float("nan"), dboxes=0,
                   rot=0, status="", ambiguous=False, ink=ink_fraction(gray_fine),
                   second_file="", second_page=0, second_hit=float("nan"), second_explained=float("nan"),
                   second_dscore=float("nan"), size=f"{rgb.shape[1]}x{rgb.shape[0]}")
        label = f"{slug(f)}_p{p + 1}"

        if row["ink"] < args.blank_ink:
            row["status"] = "blank"
        else:
            best = second = None
            ambiguous = False
            pre_rot = 0
            # SIFT is rotation invariant, but RANSAC can still lock onto the shared template on a
            # page that is upside down and heavily scribbled: retry with the scan pre-rotated.
            for r in (0, 180, 90, 270):
                g = rotate90(gray_full, r) if r else gray_full
                b, sec, amb = match_page(g, orig, feats, args.top_k, args.ambiguous_margin, args.fine_width)
                if b and (best is None or b["combined"] > best["combined"]):
                    best, second, ambiguous, pre_rot = b, sec, amb, r
                if best and best["inl"] >= args.min_inliers and best["hit"] >= args.min_hit:
                    break
            if best:
                o = best["orig"]
                row.update(original_file=o["file"], original_page=o["page"], inliers=best["inl"],
                           hit_rate=best["hit"], explained=best["explained"],
                           dscore=best.get("dscore", float("nan")), dboxes=best.get("dboxes", 0),
                           rot=(pre_rot + best["rot"]) % 360, ambiguous=ambiguous)
                if second:
                    row.update(second_file=second["orig"]["file"], second_page=second["orig"]["page"],
                               second_hit=second["hit"], second_explained=second["explained"],
                               second_dscore=second.get("dscore", float("nan")))
                ok = best["inl"] >= args.min_inliers and best["hit"] >= args.min_hit
                row["status"] = "matched" if ok else "low_confidence"
            else:
                row["status"] = "unmatched"

        if row["status"] != "matched":
            thumb = resize_width(gray_full, 600)
            cv2.imwrite(os.path.join(thumb_dir, f"{label}.png"), thumb)
            row["thumb"] = os.path.join(thumb_dir, f"{label}.png")
        else:
            row["thumb"] = ""
        rows.append(row)
        tgt = f"{slug(row['original_file'])} p{row['original_page']}" if row["original_file"] else "-"
        print(f"  [{src:7s}] {label:28s} {row['status']:14s} -> {tgt:40s} inl={row['inliers']:4d} "
              f"hit={row['hit_rate']:.3f} expl={row['explained']:.2f} dscore={row['dscore']:.2f}/{row['dboxes']}bx rot={row['rot']:3d} "
              f"ink={row['ink']:.4f}{' AMBIG' if row.get('ambiguous') else ''} ({time.time() - t:.1f}s)")

    # ----------------------------------------------------------------- write manifest
    fields = ["scan_file", "scan_page", "original_file", "original_page", "inliers", "hit_rate", "explained",
              "dscore", "dboxes", "source", "status", "ambiguous", "rot", "ink", "size",
              "second_file", "second_page", "second_hit", "second_explained", "second_dscore", "thumb"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)

    # ----------------------------------------------------------------- summary
    lines = []
    lines.append(f"scan pages: {len(rows)}   matched: {sum(r['status'] == 'matched' for r in rows)}   "
                 f"low_confidence: {sum(r['status'] == 'low_confidence' for r in rows)}   "
                 f"unmatched: {sum(r['status'] == 'unmatched' for r in rows)}   "
                 f"blank: {sum(r['status'] == 'blank' for r in rows)}")
    lines.append("")
    lines.append("coverage per original page (printer / phone answered copies):")
    for o in orig:
        n_pr = sum(1 for r in rows if r["status"] == "matched" and r["original_file"] == o["file"]
                   and r["original_page"] == o["page"] and r["source"] == "printer")
        n_ph = sum(1 for r in rows if r["status"] == "matched" and r["original_file"] == o["file"]
                   and r["original_page"] == o["page"] and r["source"] == "phone")
        flag = "" if (n_pr + n_ph) else "   <-- no answered copy"
        lines.append(f"  {slug(o['file']):40s} p{o['page']:<3d} printer={n_pr:2d} phone={n_ph:2d}{flag}")
    lines.append("")
    amb = [r for r in rows if r["status"] == "matched" and r.get("ambiguous")]
    if amb:
        lines.append("matched but ambiguous (runner-up within margin; verify visually):")
        for r in amb:
            lines.append(f"  [{r['source']}] {slug(r['scan_file'])} p{r['scan_page']:<3d} -> {slug(r['original_file'])} p{r['original_page']} "
                         f"hit={r['hit_rate']:.3f} expl={r['explained']:.2f} dscore={r['dscore']:.2f}/{r['dboxes']}bx   "
                         f"runner-up {slug(r['second_file'])} p{r['second_page']} hit={r['second_hit']:.3f} "
                         f"expl={r['second_explained']:.2f} dscore={r['second_dscore']:.2f}")
        lines.append("")
    lines.append("pages needing attention (thumbnails in thumbs/):")
    for r in rows:
        if r["status"] != "matched":
            best = f"best {slug(r['original_file'])} p{r['original_page']} inl={r['inliers']} hit={r['hit_rate']:.2f}" \
                if r["original_file"] else "no candidate"
            lines.append(f"  [{r['source']}] {slug(r['scan_file'])} p{r['scan_page']:<3d} {r['status']:14s} {best}   {r['thumb']}")
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print(f"\nmanifest: {os.path.join(args.out, 'manifest.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
