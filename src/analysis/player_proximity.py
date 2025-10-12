#!/usr/bin/env python3
"""
Compute 'near player' metrics for Spot-the-Ball predictions, restricted to model subfolders.

Reads:
  - filtered_people_image_summary.csv   (one row per image; includes image path, img_w, img_h, sport)
  - filtered_people_boxes_long.csv      (one row per kept player box; x1,y1,x2,y2, image)
  - prediction CSVs ONLY from subfolders under --pred-root:
        gemini/, gpt/, llama/, qwen/, human/
    (you can change the included set via --models)

Supports both formats:
  - AI: columns include 'image', 'cell' (e.g., "B6" or "6B")
  - Human: columns include 'image', 'row', 'col'

Outputs (to --outdir):
  - proximity_per_prediction.csv   (one row per prediction)
  - proximity_summary.csv          (grouped by sport/model/level)

Example:
  python player_proximity.py \
    --img-summary "/.../filtered_people_image_summary.csv" \
    --boxes-long  "/.../filtered_people_boxes_long.csv" \
    --pred-root   "/.../data" \
    --models      "gemini,gpt,llama,qwen,human" \
    --outdir      "/.../data"
"""

import argparse
import os
import re
import glob
import math
import numpy as np
import pandas as pd

# -----------------------
# CLI
# -----------------------

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-summary', required=True, help='filtered_people_image_summary.csv')
    ap.add_argument('--boxes-long',  required=True, help='filtered_people_boxes_long.csv')
    ap.add_argument('--pred-root',   required=True, help='Root folder containing model subfolders (gemini,gpt,llama,qwen,human)')
    ap.add_argument('--models',      default='gemini,gpt,llama,qwen,human',
                    help='Comma-separated list of model subfolders to include under --pred-root')
    ap.add_argument('--outdir',      required=True, help='Output directory')
    ap.add_argument('--rows',        type=int, default=6,  help='Grid rows (default 6)')
    ap.add_argument('--cols',        type=int, default=10, help='Grid cols (default 10)')
    ap.add_argument('--tau-near',    type=float, default=0.08, help='Point-to-box distance threshold as fraction of image diag')
    ap.add_argument('--dilate',      type=float, default=1, help='Box dilation factor for overlap metric')
    ap.add_argument('--theta',       type=float, default=0.02, help='Min overlap fraction of cell area')
    return ap.parse_args()

# -----------------------
# Utilities
# -----------------------

ROW_MAP = {ch:i for i,ch in enumerate(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}

def canonical_sport(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return None
    s = str(s).strip().lower()
    if s in ('vball','vb','volley','volleyball'): return 'volleyball'
    if s in ('bb','basket','basketball'):         return 'basketball'
    if s in ('soccer','football'):                return 'soccer'
    return s

def parse_meta_from_path(path):
    """
    Infer (level, sport) from filename like 'level2_soccer.csv'.
    """
    base = os.path.basename(path)
    m = re.search(r'level\s*([0-2])[_\- ]+([A-Za-z]+)', base, re.IGNORECASE)
    level = int(m.group(1)) if m else None
    sport = canonical_sport(m.group(2)) if m else None
    return level, sport

def image_base_with_ext(v):
    return os.path.basename(str(v))

def image_stem(v):
    b = image_base_with_ext(v)
    return os.path.splitext(b)[0]

def cell_to_rect(cell, w, h, rows=6, cols=10):
    """
    Accepts 'A5' or '5A' (case-insensitive), returns (x1,y1,x2,y2,cx,cy) in pixels.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return None
    cell = str(cell).strip().upper()
    m1 = re.match(r'^([A-Z])\s*([1-9]|10)$', cell)   # 'A5'
    m2 = re.match(r'^([1-9]|10)\s*([A-Z])$', cell)   # '5A'
    if m1:
        r = ROW_MAP[m1.group(1)]; c = int(m1.group(2)) - 1
    elif m2:
        r = ROW_MAP[m2.group(2)]; c = int(m2.group(1)) - 1
    else:
        return None
    if r < 0 or r >= rows or c < 0 or c >= cols: return None
    cw, ch = w/cols, h/rows
    x1, y1 = c*cw, r*ch
    x2, y2 = x1 + cw, y1 + ch
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    return (x1,y1,x2,y2,cx,cy)

def point_to_rect_edge_dist(px, py, b):
    x1,y1,x2,y2 = b
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)

def min_point_to_any_box(px, py, boxes):
    if boxes is None or len(boxes) == 0: return np.nan
    return float(np.min([point_to_rect_edge_dist(px,py,b) for b in boxes]))

def dilate_box(b, scale, w, h):
    x1,y1,x2,y2 = b
    cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
    bw, bh = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = max(0, cx-bw/2.0), max(0, cy-bh/2.0)
    nx2, ny2 = min(w, cx+bw/2.0), min(h, cy+bh/2.0)
    return (nx1,ny1,nx2,ny2)

def rect_overlap(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    w = max(0.0, ix2-ix1); h = max(0.0, iy2-iy1)
    return w*h

def build_image_keys(df_img):
    """
    Return:
      sizes: image_base_with_ext -> {'img_w': w, 'img_h': h}
      sport_by_image: image_base_with_ext -> canonical sport (if present)
    """
    sizes, sports = {}, {}
    for _, r in df_img.iterrows():
        base = image_base_with_ext(r['image'])
        sizes[base] = {'img_w': float(r['img_w']), 'img_h': float(r['img_h'])}
        sports[base] = canonical_sport(r.get('sport'))
    return sizes, sports

def build_boxes_map(df_box):
    """
    Map image_base_with_ext -> np.array of boxes (x1,y1,x2,y2)
    """
    boxes_by_image = {}
    if 'image' not in df_box.columns:
        raise SystemExit("filtered_people_boxes_long.csv must have an 'image' column")
    for base, sub in df_box.groupby(df_box['image'].apply(image_base_with_ext)):
        boxes = sub[['x1','y1','x2','y2']].to_numpy(dtype=float)
        boxes_by_image[base] = boxes
    return boxes_by_image

def resolve_image_key(raw_image_value, size_keys):
    """
    Try to match a prediction 'image' value to a key in sizes (basenames with ext).
    1) exact basename match
    2) if no ext, try .jpg/.png/.jpeg
    3) fallback: unique match by stem
    """
    raw = str(raw_image_value)
    base = image_base_with_ext(raw)
    if base in size_keys:
        return base
    stem = image_stem(base)
    for ext in ('.jpg', '.png', '.jpeg'):
        cand = stem + ext
        if cand in size_keys:
            return cand
    matches = [k for k in size_keys if image_stem(k) == stem]
    if len(matches) == 1:
        return matches[0]
    return None

# -----------------------
# Main
# -----------------------

def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load image summary / boxes
    df_img = pd.read_csv(args.img_summary)
    df_box = pd.read_csv(args.boxes_long)

    # Clean column names (strip spaces)
    df_img.columns = df_img.columns.str.strip()
    df_box.columns = df_box.columns.str.strip()

    if not {'image','img_w','img_h'}.issubset(set(df_img.columns)):
        raise SystemExit("img-summary must contain columns: image, img_w, img_h")

    sizes, sport_by_image = build_image_keys(df_img)
    boxes_by_image = build_boxes_map(df_box)
    size_keys = set(sizes.keys())

    # Collect prediction CSVs from model subfolders
    wanted_models = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    model_dirs = [os.path.join(args.pred_root, m) for m in wanted_models]
    pred_files = []
    for m, d in zip(wanted_models, model_dirs):
        if not os.path.isdir(d):
            print(f'[WARN] model folder not found: {d}')
            continue
        found = sorted(glob.glob(os.path.join(d, '**', '*.csv'), recursive=True))
        print(f'[INFO] {m}: {len(found)} CSV files')
        pred_files.extend([(m, pf) for pf in found])

    if not pred_files:
        raise SystemExit('No prediction CSVs found under the specified model folders.')

    per_pred_rows = []
    unmatched_images = 0
    total_preds = 0

    for folder_model, pf in pred_files:
        try:
            dfp = pd.read_csv(pf)
        except Exception as e:
            print(f'[WARN] Could not read {pf}: {e}')
            continue

        # normalize cols
        dfp.columns = dfp.columns.str.strip()
        if 'image' not in dfp.columns:
            print(f'[WARN] Skipping {pf}: no "image" column.')
            continue

        # infer (level, sport) from file name; prefer CSV columns if present
        file_level, file_sport = parse_meta_from_path(pf)

        # Fill missing meta
        if 'model' not in dfp.columns or dfp['model'].isna().all():
            dfp['model'] = folder_model
        if 'level' not in dfp.columns or dfp['level'].isna().all():
            dfp['level'] = file_level
        if 'sport' not in dfp.columns or dfp['sport'].isna().all():
            dfp['sport'] = file_sport

        # iterate predictions
        for _, r in dfp.iterrows():
            total_preds += 1
            raw_img = r['image']
            key = resolve_image_key(raw_img, size_keys)
            if key is None:
                unmatched_images += 1
                continue

            w = float(sizes[key]['img_w']); h = float(sizes[key]['img_h'])
            diag = math.hypot(w, h)

            # Determine predicted cell (AI 'cell' OR human 'row','col')
            pred_cell = None
            if 'cell' in dfp.columns and pd.notna(r.get('cell')):
                pred_cell = str(r.get('cell'))
            elif 'row' in dfp.columns and 'col' in dfp.columns and pd.notna(r.get('row')) and pd.notna(r.get('col')):
                try:
                    row = int(r.get('row')); col = int(r.get('col'))
                    pred_cell = f"{chr(ord('A') + row - 1)}{col}"
                except Exception:
                    pred_cell = None

            rect = cell_to_rect(pred_cell, w, h, rows=args.rows, cols=args.cols)
            if rect is None:
                continue
            x1,y1,x2,y2,cx,cy = rect
            cell_area = (x2-x1)*(y2-y1)

            boxes = boxes_by_image.get(key, np.zeros((0,4), dtype=float))

            # Metric 1: distance to nearest player
            d_min = min_point_to_any_box(cx, cy, boxes)
            near_dist = (not np.isnan(d_min)) and (d_min <= args.tau_near * diag)
            d_min_norm = (d_min/diag) if (not np.isnan(d_min) and diag>0) else np.nan

            # Metric 2: overlap with dilated box
            if len(boxes) > 0:
                dilated = [dilate_box(tuple(b), args.dilate, w, h) for b in boxes]
                ovlp_area = max([rect_overlap((x1,y1,x2,y2), db) for db in dilated], default=0.0)
            else:
                ovlp_area = 0.0
            near_overlap = (cell_area > 0) and ((ovlp_area / cell_area) >= args.theta)

            per_pred_rows.append({
                'image_base': key,
                'sport': canonical_sport(r.get('sport')) or sport_by_image.get(key),
                'model': str(r.get('model')).lower() if pd.notna(r.get('model')) else folder_model,
                'level': int(r.get('level')) if pd.notna(r.get('level')) else None,
                'pred_cell': pred_cell,
                'd_min_px': d_min,
                'd_min_norm': d_min_norm,
                'near_dist': bool(near_dist),
                'overlap_area_px': ovlp_area,
                'cell_area_px': cell_area,
                'near_overlap': bool(near_overlap),
                'tau_near': args.tau_near,
                'dilate': args.dilate,
                'theta': args.theta
            })

    if unmatched_images:
        print(f'[INFO] Unmatched predictions (image name not found in image summary): {unmatched_images} of {total_preds}')

    if not per_pred_rows:
        raise SystemExit('No predictions matched to images; nothing to write.')

    df_pred = pd.DataFrame(per_pred_rows)
    df_pred['sport'] = df_pred['sport'].apply(canonical_sport)

    per_path = os.path.join(args.outdir, 'proximity_per_prediction.csv')
    df_pred.to_csv(per_path, index=False)

    summary = (df_pred
               .groupby(['sport','model','level'], dropna=False)
               .agg(n=('image_base','count'),
                    near_dist_rate=('near_dist','mean'),
                    near_overlap_rate=('near_overlap','mean'),
                    median_d_norm=('d_min_norm','median'))
               .reset_index()
               .sort_values(['sport','model','level']))
    sum_path = os.path.join(args.outdir, 'proximity_summary.csv')
    summary.to_csv(sum_path, index=False)

    print(f'Wrote:\n- {per_path} ({len(df_pred)} rows)\n- {sum_path} ({len(summary)} rows)')
    print('Done.')

if __name__ == '__main__':
    main()
