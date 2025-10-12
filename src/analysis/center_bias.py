import os
import re
import json
import math
import numpy as np
import pandas as pd
from itertools import product
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact, chi2_contingency, spearmanr, norm
from scipy.optimize import linprog  # for exact EMD on the grid with a cost matrix
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

# --------------------
# Configuration
# --------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = DATA_DIR / "analysis"
VISUALS_DIR = ANALYSIS_DIR / "visuals"

# --------------------
# Constants (adapt if needed)
# --------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
GRID_ROWS = 6
GRID_COLS = 10
SPORTS = ['soccer', 'vball', 'bb']        # use 'bb' for basketball in file paths, label later
LEVELS = [0, 1, 2]
MODELS = ['gpt', 'gemini', 'qwen', 'llama']
HUMAN = 'human'

# Ground truth files (found in current structure)
GROUND_TRUTH = {
    'soccer': DATA_DIR / "images" / "soccer" / "soccer_gt.csv",
    'vball':  DATA_DIR / "images" / "volleyball" / "vball_gt.csv",
    'bb':     DATA_DIR / "images" / "basketball" / "bb_gt.csv"
}

# Human data (if available - these paths may need to be updated)
HUMAN_DATA = {
    'soccer': DATA_DIR / "humans" / "soccer_human.csv",
    'vball':  DATA_DIR / "humans" / "vball_human.csv",
    'bb':     DATA_DIR / "humans" / "bb_human.csv"
}

# Output directory for analysis results
OUTPUT_DIR = ANALYSIS_DIR / "center_bias"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def grid_cell_center(row, col):
    if 1 <= row <= GRID_ROWS and 1 <= col <= GRID_COLS:
        x = (col - 0.5) * (IMAGE_WIDTH / GRID_COLS)
        y = (row - 0.5) * (IMAGE_HEIGHT / GRID_ROWS)
        return x, y
    raise ValueError(f'Out of bounds: r{row}, c{col}')

def alpha_to_rc(cell):
    """A1..F10 to (row, col)"""
    cell = str(cell).strip().upper()
    row = ord(cell[0]) - ord('A') + 1
    col = int(cell[1:])
    if not (1 <= row <= GRID_ROWS and 1 <= col <= GRID_COLS):
        raise ValueError(f'Out of bounds cell {cell}')
    return row, col

def rc_to_alpha(row, col):
    return f'{chr(ord("A")+row-1)}{col}'

def both_formats_to_rc(cell):
    """Accept A7 or 3C and return (row, col)."""
    s = str(cell).strip().upper()
    if not s:
        raise ValueError('empty cell')
    
    # Validate format: should be either letter+number or number+letter
    if s[0].isalpha():  # A7 format
        # Check if the rest is numeric
        if not s[1:].isdigit():
            raise ValueError(f'Invalid cell format: {cell} - expected letter followed by number (e.g., A7)')
        return alpha_to_rc(s)
    elif s[-1].isalpha():  # 3C format
        # Check if the first part is numeric
        if not s[:-1].isdigit():
            raise ValueError(f'Invalid cell format: {cell} - expected number followed by letter (e.g., 3C)')
        row = int(s[:-1])
        col = ord(s[-1]) - ord('A') + 1
        if not (1 <= row <= GRID_ROWS and 1 <= col <= GRID_COLS):
            raise ValueError(f'Out of bounds cell {cell}')
        return row, col
    else:
        raise ValueError(f'Invalid cell format: {cell} - must be letter+number (e.g., A7) or number+letter (e.g., 3C)')

def normalize_image_id(val):
    """Return int image id from things like '12', '12.jpg', 'soccer_0012.png'."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r'\.(png|jpg|jpeg)$', '', s, flags=re.I)
    m = re.findall(r'\d+', s)
    return int(m[-1]) if m else None

def central_window_indices():
    """Define a central block, adjust if you want stricter center."""
    rows = range(2, 5)   # 2..4
    cols = range(3, 8)   # 3..7
    return { (r-1)*GRID_COLS + (c-1) for r in rows for c in cols }

def all_cell_centers_matrix():
    centers = []
    for r in range(1, GRID_ROWS+1):
        for c in range(1, GRID_COLS+1):
            centers.append(grid_cell_center(r, c))
    return np.array(centers)  # shape (60, 2)

CELL_CENTERS = all_cell_centers_matrix()

def pairwise_cost_matrix():
    """60x60 matrix of Euclidean distances between cell centers."""
    diffs = CELL_CENTERS[:, None, :] - CELL_CENTERS[None, :, :]
    return np.sqrt((diffs**2).sum(axis=2))
COST = pairwise_cost_matrix()

def truth_indices_from_set(cells):
    idxs = []
    for cell in cells:
        r, c = alpha_to_rc(cell)
        idxs.append((r-1)*GRID_COLS + (c-1))
    return sorted(idxs)

# --------------------
# Data loading to a unified long table of predictions
# --------------------
def load_ground_truth_df(sport):
    gt = pd.read_csv(GROUND_TRUTH[sport]).pipe(clean_columns)
    gt['image'] = pd.to_numeric(gt['image'], errors='coerce').astype('Int64')
    gt = gt.dropna(subset=['image'])
    # Normalize and store lists of truth cells
    gt['truth_cells'] = gt['cell'].astype(str).apply(lambda s: [c.strip().upper() for c in s.split(';')] if ';' in s else [s.strip().upper()])
    # Derived features
    gt['truth_indices'] = gt['truth_cells'].apply(truth_indices_from_set)
    # Truth center for distance to image center
    def mean_truth_center(cells):
        pts = [grid_cell_center(*alpha_to_rc(c)) for c in cells]
        return np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    gt[['truth_x', 'truth_y']] = gt['truth_cells'].apply(lambda cs: pd.Series(mean_truth_center(cs)))
    img_cx, img_cy = IMAGE_WIDTH/2, IMAGE_HEIGHT/2
    gt['dist_to_center'] = np.sqrt((gt['truth_x']-img_cx)**2 + (gt['truth_y']-img_cy)**2)
    gt['cells_spanned'] = gt['truth_cells'].apply(len)
    return gt[['image', 'truth_cells', 'truth_indices', 'truth_x', 'truth_y', 'dist_to_center', 'cells_spanned']]

def load_human_predictions(sport):
    path = HUMAN_DATA[sport]
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path).pipe(clean_columns)
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}")
        return pd.DataFrame()
        
    if 'image' not in df.columns:
        return pd.DataFrame()
    df['image_id'] = pd.to_numeric(df['image'], errors='coerce').astype('Int64')
    # Best effort row col names
    row_cands = ['row', 'Row', 'row_pred', 'r']
    col_cands = ['col', 'Col', 'col_pred', 'c']
    row_col = next((c for c in row_cands if c in df.columns), None)
    col_col = next((c for c in col_cands if c in df.columns), None)
    if not row_col or not col_col:
        return pd.DataFrame()
    df = df.dropna(subset=['image_id', row_col, col_col])
    df['row'] = pd.to_numeric(df[row_col], errors='coerce').astype('Int64')
    df['col'] = pd.to_numeric(df[col_col], errors='coerce').astype('Int64')
    df = df.dropna(subset=['row','col'])
    df['pred_cell'] = df.apply(lambda r: rc_to_alpha(int(r['row']), int(r['col'])), axis=1)
    df['model'] = HUMAN
    df['level'] = 0
    return df[['image_id','model','level','pred_cell']]

def load_model_predictions(sport, model, level):
    path = RESULTS_DIR / model / f"{sport}_level{level}.csv"
    if not path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path).pipe(clean_columns)
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}")
        return pd.DataFrame()
        
    if 'image' not in df.columns:
        return pd.DataFrame()
    df['image_id'] = df['image'].apply(normalize_image_id).astype('Int64')
    # find cell column
    pred_col = next((c for c in ['cell','pred_cell','prediction','Cell','CELL'] if c in df.columns), None)
    if pred_col is None:
        return pd.DataFrame()
    df[pred_col] = (df[pred_col].astype(str)
                               .str.strip()
                               .str.replace(r'\s+','', regex=True)
                               .str.upper())
    df = df.dropna(subset=['image_id', pred_col])
    df = df[df[pred_col] != '']
    df['pred_cell'] = df[pred_col]
    df['model'] = model
    df['level'] = level
    return df[['image_id','model','level','pred_cell']]

def build_long_predictions():
    rows = []
    for sport in SPORTS:
        # human
        h = load_human_predictions(sport)
        if not h.empty:
            h['sport'] = sport
            rows.append(h)
        # models
        for model, level in product(MODELS, LEVELS):
            m = load_model_predictions(sport, model, level)
            if not m.empty:
                m['sport'] = sport
                rows.append(m)
    if not rows:
        return pd.DataFrame()
    preds = pd.concat(rows, ignore_index=True)
    return preds[['sport','image_id','model','level','pred_cell']]

# --------------------
# Metric computations
# --------------------
def per_group_accuracy(preds, gt):
    """Accuracy per (sport, model, level)."""
    # Build quick lookup of truth cell sets
    truth_map = dict(zip(gt['image'], gt['truth_cells']))
    def correct_row(row):
        cells = truth_map.get(int(row['image_id']))
        if not cells: 
            return np.nan
        return 1.0 if row['pred_cell'] in cells else 0.0
    tmp = preds.copy()
    tmp['correct'] = tmp.apply(correct_row, axis=1)
    tmp = tmp.dropna(subset=['correct'])
    grp = tmp.groupby(['sport','model','level'])['correct']
    out = grp.agg(n='count', acc='mean')
    out['se'] = np.sqrt(out['acc']*(1-out['acc'])/out['n']).replace([np.inf, np.nan], 0.0)
    return out.reset_index()

def imagewise_prediction_distributions(preds, gt):
    """Return dict keyed by (sport, model, level, image) with prob vectors over 60 cells."""
    # map truth indices too
    truth_idx_map = dict(zip(gt['image'], gt['truth_indices']))
    dist = {}
    invalid_cells = []
    
    for (sport, model, level, image), g in preds.groupby(['sport','model','level','image_id']):
        counts = np.zeros(GRID_ROWS*GRID_COLS, dtype=np.float64)
        for cell in g['pred_cell']:
            # Skip NaN values
            if pd.isna(cell) or str(cell).strip().upper() in ['NAN', 'NAN', '']:
                continue
                
            try:
                r, c = both_formats_to_rc(cell)
                idx = (r-1)*GRID_COLS + (c-1)
                counts[idx] += 1.0
            except ValueError as e:
                # Log invalid cells for debugging
                invalid_cells.append({
                    'sport': sport,
                    'model': model,
                    'level': level,
                    'image': image,
                    'cell': cell,
                    'error': str(e)
                })
                continue
        
        total = counts.sum()
        if total <= 0:
            continue
        p = counts / total
        truth_idxs = truth_idx_map.get(int(image), [])
        dist[(sport, model, level, int(image))] = {'p': p, 'truth': truth_idxs}
    
    # Log invalid cells if any found
    if invalid_cells:
        print(f"Warning: Found {len(invalid_cells)} invalid cell values:")
        for ic in invalid_cells[:10]:  # Show first 10
            print(f"  {ic['sport']} {ic['model']} level{ic['level']} image{ic['image']}: '{ic['cell']}' - {ic['error']}")
        if len(invalid_cells) > 10:
            print(f"  ... and {len(invalid_cells) - 10} more")
    
    return dist

def wasserstein_to_truth(p, truth_idxs, cost_matrix=COST):
    """
    Exact EMD to a uniform distribution over truth cells using a linear program.
    Source bins: 60 with masses p_j.
    Target bins: |G| with masses 1/|G|.
    Variables: 60*|G| flows.
    """
    if len(truth_idxs) == 0:
        return np.nan
    J = len(p)
    K = len(truth_idxs)

    # Flatten cost for variables f_{j,k}
    c = []
    for j in range(J):
        for k in range(K):
            c.append(cost_matrix[j, truth_idxs[k]])
    c = np.array(c)

    # Equality constraints: 
    # 1) For each source j: sum_k f_{j,k} = p_j
    A_eq = np.zeros((J + K, J*K))
    b_eq = np.zeros(J + K)
    # source constraints
    for j in range(J):
        A_eq[j, j*K:(j+1)*K] = 1.0
        b_eq[j] = p[j]
    # target constraints: for each k: sum_j f_{j,k} = 1/K
    for k in range(K):
        for j in range(J):
            A_eq[J+k, j*K + k] = 1.0
        b_eq[J+k] = 1.0 / K

    bounds = [(0, None)] * (J*K)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return np.nan
    return res.fun

def mean_euclidean_distance_to_truth(pred_cells, truth_cells):
    """Average over predictions of distance to nearest truth center."""
    truth_xy = [grid_cell_center(*alpha_to_rc(c)) for c in truth_cells]
    dists = []
    invalid_cells = []
    
    for cell in pred_cells:
        # Skip NaN values
        if pd.isna(cell) or str(cell).strip().upper() in ['NAN', 'NAN', '']:
            continue
            
        try:
            r, c = both_formats_to_rc(cell)
            x, y = grid_cell_center(r, c)
            d = min(np.hypot(x - xt, y - yt) for xt, yt in truth_xy)
            dists.append(d)
        except ValueError as e:
            # Log invalid cells for debugging
            invalid_cells.append({'cell': cell, 'error': str(e)})
            continue
    
    if invalid_cells:
        print(f"Warning: Found {len(invalid_cells)} invalid cell values in distance calculation:")
        for ic in invalid_cells[:5]:  # Show first 5
            print(f"  '{ic['cell']}' - {ic['error']}")
        if len(invalid_cells) > 5:
            print(f"  ... and {len(invalid_cells) - 5} more")
    
    if not dists:
        return np.nan
    return float(np.mean(dists))

def per_group_distances(preds, gt):
    """
    Compute per image Wasserstein and Euclidean errors then aggregate per group.
    """
    gt_map_cells = dict(zip(gt['image'], gt['truth_cells']))
    dists = []
    # prepare distributions
    dist_map = imagewise_prediction_distributions(preds, gt)
    for (sport, model, level, image), rec in dist_map.items():
        p = rec['p']; truth = rec['truth']
        w = wasserstein_to_truth(p, truth)
        # Euclidean mean over samples
        pred_cells = preds[(preds['sport']==sport) & (preds['model']==model) & 
                           (preds['level']==level) & (preds['image_id']==image)]['pred_cell'].tolist()
        
        # Check if image exists in ground truth
        if image not in gt_map_cells:
            continue
            
        e = mean_euclidean_distance_to_truth(pred_cells, gt_map_cells[image])
        dists.append({'sport':sport,'model':model,'level':level,'image':image,'wasserstein':w,'euclid':e})
    df = pd.DataFrame(dists)
    agg = df.groupby(['sport','model','level']).agg(
        n=('image','count'),
        wasserstein_mean=('wasserstein','mean'),
        wasserstein_std=('wasserstein','std'),
        euclid_mean=('euclid','mean'),
        euclid_std=('euclid','std')
    ).reset_index()
    return df, agg

def bootstrap_ci(values, B=1000, func=np.mean, seed=0):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    stats = []
    n = len(values)
    for _ in range(B):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(func(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)

def add_bootstrap_to_group_table(df, colname, B=1000):
    rows = []
    for _, r in df.iterrows():
        # You need the image wise table to compute per group CI
        rows.append(r.to_dict())
    return df  # place holder if you want fast aggregation only

# --------------------
# Tests and effect sizes
# --------------------
def human_model_tables(acc_table):
    """Return 2x2 tables and p values for human vs each model per sport and level."""
    out = []
    # expand accuracy table to counts; we need raw correct counts
    # To get counts, recompute from predictions to be safe
    return out

def cohen_h_for_props(p1, p2):
    return 2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2))

def prompt_deltas(acc_table):
    """Level deltas per model and sport."""
    rows = []
    for sport in SPORTS:
        for model in MODELS:
            sub = acc_table[(acc_table['sport']==sport) & (acc_table['model']==model)]
            if sub.empty: continue
            acc0 = sub.loc[sub['level']==0, 'acc']
            for lvl in [1,2]:
                accL = sub.loc[sub['level']==lvl, 'acc']
                if not acc0.empty and not accL.empty:
                    rows.append({'sport':sport,'model':model,'level':lvl,'delta': float(accL.values[0]-acc0.values[0])})
    return pd.DataFrame(rows)

def sport_rank_correlation(acc_table, humans):
    """Spearman rho between human and each model on [soccer, vball, bb]."""
    rows = []
    for model in MODELS:
        v_model = []
        v_human = []
        for sport in SPORTS:
            hm = humans[(humans['sport']==sport)]
            md = acc_table[(acc_table['sport']==sport) & (acc_table['model']==model)]
            if hm.empty or md.empty: continue
            # choose one level or average across levels; here average
            v_human.append(hm['acc'].mean())
            v_model.append(md['acc'].mean())
        if len(v_model) == len(SPORTS):
            rho, p = spearmanr(v_human, v_model)
            rows.append({'model':model,'spearman_rho':rho, 'p_value':p})
    return pd.DataFrame(rows)

# --------------------
# Center bias and entropy
# --------------------

def center_bias(preds, gt):
    central = central_window_indices()
    # prior mass from truth occupancy across images
    J = GRID_ROWS*GRID_COLS
    truth_prior = np.zeros(J)
    for idxs in gt['truth_indices']:
        for j in idxs:
            truth_prior[j] += 1
    truth_prior = truth_prior / truth_prior.sum() if truth_prior.sum()>0 else truth_prior

    results = []
    for (sport, model, level), g in preds.groupby(['sport','model','level']):
        counts = np.zeros(J)
        for cell in g['pred_cell']:
            # Skip NaN values
            if pd.isna(cell) or str(cell).strip().upper() in ['NAN', 'NAN', '']:
                continue
                
            try:
                r, c = both_formats_to_rc(cell)
                j = (r-1)*GRID_COLS + (c-1)
                counts[j] += 1.0
            except ValueError as e:
                # Skip invalid cells
                continue
                
        if counts.sum() == 0: 
            continue
        p = counts / counts.sum()
        central_mass = p[list(central)].sum()
        prior_central = truth_prior[list(central)].sum()
        ratio = (central_mass / prior_central) if prior_central > 0 else np.nan
        entropy = -(p[p>0] * np.log(p[p>0])).sum()
        normalized_entropy = entropy / np.log(J)

        results.append({'sport':sport,'model':model,'level':level,
                        'central_mass': central_mass,
                        'prior_central': prior_central,
                        'central_ratio': ratio,
                        'entropy': entropy,
                        'normalized_entropy': normalized_entropy
                        })
    return pd.DataFrame(results)

def center_bias_by_model(preds, gt):
    """Center bias and entropy per model (across all sports and levels)."""
    central = central_window_indices()
    # prior mass from truth occupancy across images
    J = GRID_ROWS*GRID_COLS
    truth_prior = np.zeros(J)
    for idxs in gt['truth_indices']:
        for j in idxs:
            truth_prior[j] += 1
    truth_prior = truth_prior / truth_prior.sum() if truth_prior.sum()>0 else truth_prior

    results = []
    for model, g in preds.groupby(['model']):
        counts = np.zeros(J)
        for cell in g['pred_cell']:
            # Skip NaN values
            if pd.isna(cell) or str(cell).strip().upper() in ['NAN', 'NAN', '']:
                continue
                
            try:
                r, c = both_formats_to_rc(cell)
                j = (r-1)*GRID_COLS + (c-1)
                counts[j] += 1.0
            except ValueError as e:
                # Skip invalid cells
                continue
                
        if counts.sum() == 0: 
            continue
        p = counts / counts.sum()
        central_mass = p[list(central)].sum()
        prior_central = truth_prior[list(central)].sum()
        ratio = (central_mass / prior_central) if prior_central > 0 else np.nan
        entropy = -(p[p>0] * np.log(p[p>0])).sum()
        normalized_entropy = entropy / np.log(J)
        results.append({'model': model,
                        'central_mass': central_mass,
                        'prior_central': prior_central,
                        'central_ratio': ratio,
                        'entropy': entropy,
                        'normalized_entropy': normalized_entropy})
    return pd.DataFrame(results)

def entropy_by_model(preds):
    """Entropy per model (across all sports, levels, and images)."""
    results = []
    for model, g in preds.groupby(['model']):
        counts = Counter(g['pred_cell'])
        # Remove NaN values
        counts = {k: v for k, v in counts.items() 
                 if not (pd.isna(k) or str(k).strip().upper() in ['NAN', 'NAN', ''])}
        
        total = sum(counts.values())
        if total == 0:
            continue
            
        p = np.array(list(counts.values()), dtype=float)/total
        ent = -(p*np.log(p)).sum()
        results.append({'model': model, 'entropy': ent, 'total_predictions': total})
    return pd.DataFrame(results)

def near_miss_rates(preds, gt, radius_cells=1):
    """Share of wrong predictions that are within a Manhattan radius in grid units to any truth cell."""
    def manhattan(a, b):
        ra, ca = divmod(a, GRID_COLS)
        rb, cb = divmod(b, GRID_COLS)
        return abs(ra-rb) + abs(ca-cb)
    truth_map = dict(zip(gt['image'], gt['truth_indices']))
    rows = []
    for (sport, model, level), g in preds.groupby(['sport','model','level']):
        wrong = 0
        near = 0
        for _, r in g.iterrows():
            truth = truth_map.get(int(r['image_id']), [])
            if not truth:
                continue
                
            # Skip NaN values
            if pd.isna(r['pred_cell']) or str(r['pred_cell']).strip().upper() in ['NAN', 'NAN', '']:
                continue
                
            try:
                pr, pc = both_formats_to_rc(r['pred_cell'])
                pidx = (pr-1)*GRID_COLS + (pc-1)
                if pidx in truth:
                    continue
                wrong += 1
                if any(manhattan(pidx, t) <= radius_cells for t in truth):
                    near += 1
            except ValueError as e:
                # Skip invalid cells
                continue
                
        if wrong > 0:
            rows.append({'sport':sport,'model':model,'level':level,
                         'wrong':wrong,'near':near,'near_rate': near/wrong})
    return pd.DataFrame(rows)

# --------------------
# Agreement and calibration
# --------------------
def per_image_entropy(preds):
    rows = []
    for (sport, model, level, image), g in preds.groupby(['sport','model','level','image_id']):
        counts = Counter(g['pred_cell'])
        total = sum(counts.values())
        p = np.array(list(counts.values()), dtype=float)/total
        ent = -(p*np.log(p)).sum()
        rows.append({'sport':sport,'model':model,'level':level,'image':int(image),'entropy':ent})
    return pd.DataFrame(rows)

def brier_score(preds, gt):
    """Convert sample counts to a 60 class probability and compute mean squared error to truth one hot over any truth cell (uniform across truth cells)."""
    truth_map = dict(zip(gt['image'], gt['truth_indices']))
    rows = []
    for (sport, model, level, image), g in preds.groupby(['sport','model','level','image_id']):
        J = GRID_ROWS*GRID_COLS
        counts = np.zeros(J)
        for cell in g['pred_cell']:
            # Skip NaN values
            if pd.isna(cell) or str(cell).strip().upper() in ['NAN', 'NAN', '']:
                continue
                
            try:
                r, c = both_formats_to_rc(cell)
                j = (r-1)*GRID_COLS + (c-1)
                counts[j] += 1.0
            except ValueError as e:
                # Skip invalid cells
                continue
                
        if counts.sum()==0: 
            continue
        p = counts / counts.sum()
        truth = truth_map.get(int(image), [])
        if not truth:
            continue
        q = np.zeros(J)
        for t in truth:
            q[t] += 1.0/len(truth)
        bs = ((p - q)**2).mean()
        rows.append({'sport':sport,'model':model,'level':level,'image':int(image),'brier':bs})
    df = pd.DataFrame(rows)
    return df, df.groupby(['sport','model','level'])['brier'].mean().reset_index(name='brier_mean')

# --------------------
# Pipelines
# --------------------
def run_all_metrics():
    all_tables = {}

    metrics_rows = []

    master = []
    for sport in SPORTS:
        gt = load_ground_truth_df(sport)
        preds = []

        # human
        hp = load_human_predictions(sport)
        if not hp.empty:
            preds.append(hp)
        # models
        for model, level in product(MODELS, LEVELS):
            mp = load_model_predictions(sport, model, level)
            if not mp.empty:
                preds.append(mp)
        if not preds:
            continue
        pred_df = pd.concat(preds, ignore_index=True)
        pred_df['sport'] = sport

        # Accuracy
        acc = per_group_accuracy(pred_df, gt)
        acc['sport'] = sport
        master.append(acc)

        # Distances
        img_dists, group_dists = per_group_distances(pred_df, gt)

        # Prompting deltas
        deltas = prompt_deltas(acc)

        # Center bias
        ccenter = center_bias(pred_df, gt)

        # Center bias by model (across all sports and levels)
        ccenter_model = center_bias_by_model(pred_df, gt)
        
        # Entropy by model (across all sports and levels)
        ent_model = entropy_by_model(pred_df)

        # Near misses
        nm1 = near_miss_rates(pred_df, gt, radius_cells=1)
        nm2 = near_miss_rates(pred_df, gt, radius_cells=2)

        # Entropy
        ent = per_image_entropy(pred_df)

        # Brier
        brier_img, brier_grp = brier_score(pred_df, gt)

        # Save per sport artifacts
        acc.to_csv(OUTPUT_DIR / f'acc_{sport}.csv', index=False)
        img_dists.to_csv(OUTPUT_DIR / f'dist_image_{sport}.csv', index=False)
        group_dists.to_csv(OUTPUT_DIR / f'dist_group_{sport}.csv', index=False)
        deltas.to_csv(OUTPUT_DIR / f'prompt_deltas_{sport}.csv', index=False)
        ccenter.to_csv(OUTPUT_DIR / f'center_bias_{sport}.csv', index=False)
        nm1.to_csv(OUTPUT_DIR / f'near_miss_r1_{sport}.csv', index=False)
        nm2.to_csv(OUTPUT_DIR / f'near_miss_r2_{sport}.csv', index=False)
        ent.to_csv(OUTPUT_DIR / f'entropy_image_{sport}.csv', index=False)
        brier_img.to_csv(OUTPUT_DIR / f'brier_image_{sport}.csv', index=False)
        brier_grp.to_csv(OUTPUT_DIR / f'brier_group_{sport}.csv', index=False)

    if not master:
        print('No predictions found.')
        return

    acc_all = pd.concat(master, ignore_index=True)
    acc_all.to_csv(OUTPUT_DIR / 'acc_all.csv', index=False)

    # Print overall accuracy summary
    print_overall_accuracy_summary(acc_all)

    # Human vs model gaps and rank correlations
    humans = acc_all[acc_all['model']==HUMAN]
    nonhum = acc_all[acc_all['model']!=HUMAN]

    # Gaps table
    gaps = []
    for _, r in nonhum.iterrows():
        hm = humans[(humans['sport']==r['sport']) & (humans['level']==0)]
        if hm.empty: 
            continue
        gaps.append({
            'sport': r['sport'],
            'model': r['model'],
            'level': int(r['level']),
            'human_acc': float(hm['acc'].values[0]),
            'model_acc': float(r['acc']),
            'abs_gap': float(hm['acc'].values[0] - r['acc']),
            'rel_risk': float(r['acc'] / hm['acc'].values[0]) if hm['acc'].values[0]>0 else np.nan,
            'cohen_h': cohen_h_for_props(float(hm['acc'].values[0]), float(r['acc']))
        })
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(OUTPUT_DIR / 'human_model_gaps.csv', index=False)

    # Sport rank correlations
    rho_df = sport_rank_correlation(acc_all[acc_all['model']!=HUMAN], humans)
    rho_df.to_csv(OUTPUT_DIR / 'sport_rank_rho.csv', index=False)

    # Calculate model-level metrics across all sports and levels
    all_preds = []
    for sport in SPORTS:
        gt = load_ground_truth_df(sport)
        preds = []

        # human
        hp = load_human_predictions(sport)
        if not hp.empty:
            preds.append(hp)
        # models
        for model, level in product(MODELS, LEVELS):
            mp = load_model_predictions(sport, model, level)
            if not mp.empty:
                preds.append(mp)
        if not preds:
            continue
        pred_df = pd.concat(preds, ignore_index=True)
        pred_df['sport'] = sport
        all_preds.append(pred_df)
    
    if all_preds:
        combined_preds = pd.concat(all_preds, ignore_index=True)
        
        # Model-level center bias and entropy
        ccenter_all_models = center_bias_by_model(combined_preds, load_ground_truth_df(SPORTS[0]))  # Use any sport's GT for prior
        ent_all_models = entropy_by_model(combined_preds)
        
        # Save model-level results
        ccenter_all_models.to_csv(OUTPUT_DIR / 'center_bias_all_models.csv', index=False)
        ent_all_models.to_csv(OUTPUT_DIR / 'entropy_all_models.csv', index=False)
        
        print(f"Model-level metrics saved:")
        print(f"  Center bias: {OUTPUT_DIR / 'center_bias_all_models.csv'}")
        print(f"  Entropy: {OUTPUT_DIR / 'entropy_all_models.csv'}")

    print('All metrics computed and saved.')

def print_overall_accuracy_summary(acc_all):
    """Print overall accuracy summary across all sports and levels with confidence intervals."""
    print("\n" + "="*80)
    print("OVERALL ACCURACY SUMMARY (across all sports and levels)")
    print("="*80)
    
    def calculate_confidence_interval(accuracies, sample_sizes, confidence_level=0.95):
        """Calculate confidence interval for weighted average accuracy."""
        # Calculate weighted average
        weighted_acc = np.average(accuracies, weights=sample_sizes)
        
        # Calculate weighted standard error
        # For proportions, SE = sqrt(p*(1-p)/n) where n is effective sample size
        effective_n = sum(sample_sizes)
        se = np.sqrt(weighted_acc * (1 - weighted_acc) / effective_n)
        
        # Calculate confidence interval
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * se
        
        lower_ci = max(0, weighted_acc - margin_of_error)
        upper_ci = min(1, weighted_acc + margin_of_error)
        
        return weighted_acc, se, lower_ci, upper_ci
    
    # Calculate overall accuracy for each model (across all sports and levels)
    overall_by_model = []
    for model in acc_all['model'].unique():
        model_data = acc_all[acc_all['model'] == model]
        accuracies = model_data['acc'].values
        sample_sizes = model_data['n'].values
        
        acc_mean, se, lower_ci, upper_ci = calculate_confidence_interval(accuracies, sample_sizes)
        
        overall_by_model.append({
            'model': model,
            'total_predictions': sum(sample_sizes),
            'overall_accuracy': acc_mean,
            'overall_accuracy_pct': acc_mean * 100,
            'standard_error': se,
            'lower_ci_95': lower_ci * 100,
            'upper_ci_95': upper_ci * 100
        })
    
    overall_by_model = pd.DataFrame(overall_by_model)
    
    print("\nOverall accuracy by model (across all sports and levels):")
    print("95% Confidence Intervals:")
    print(overall_by_model.round(3))
    
    # Calculate overall accuracy by sport (across all models and levels)
    overall_by_sport = []
    for sport in acc_all['sport'].unique():
        sport_data = acc_all[acc_all['sport'] == sport]
        accuracies = sport_data['acc'].values
        sample_sizes = sport_data['n'].values
        
        acc_mean, se, lower_ci, upper_ci = calculate_confidence_interval(accuracies, sample_sizes)
        
        overall_by_sport.append({
            'sport': sport,
            'total_predictions': sum(sample_sizes),
            'overall_accuracy': acc_mean,
            'overall_accuracy_pct': acc_mean * 100,
            'standard_error': se,
            'lower_ci_95': lower_ci * 100,
            'upper_ci_95': upper_ci * 100
        })
    
    overall_by_sport = pd.DataFrame(overall_by_sport)
    
    print("\nOverall accuracy by sport (across all models and levels):")
    print("95% Confidence Intervals:")
    print(overall_by_sport.round(3))
    
    # Calculate overall accuracy by level (across all sports and models)
    overall_by_level = []
    for level in acc_all['level'].unique():
        level_data = acc_all[acc_all['level'] == level]
        accuracies = level_data['acc'].values
        sample_sizes = level_data['n'].values
        
        acc_mean, se, lower_ci, upper_ci = calculate_confidence_interval(accuracies, sample_sizes)
        
        overall_by_level.append({
            'level': level,
            'total_predictions': sum(sample_sizes),
            'overall_accuracy': acc_mean,
            'overall_accuracy_pct': acc_mean * 100,
            'standard_error': se,
            'lower_ci_95': lower_ci * 100,
            'upper_ci_95': upper_ci * 100
        })
    
    overall_by_level = pd.DataFrame(overall_by_level)
    
    print("\nOverall accuracy by level (across all sports and models):")
    print("95% Confidence Intervals:")
    print(overall_by_level.round(3))
    
    # Create summary table with confidence intervals
    print("\n" + "="*120)
    print("SUMMARY TABLE: Accuracy by Model, Sport, and Level (%) with 95% Confidence Intervals")
    print("="*120)
    
    # Create pivot table for easy viewing
    pivot_table = acc_all.pivot_table(
        index='model', 
        columns='sport', 
        values='acc',
        aggfunc=lambda x: np.average(x, weights=acc_all.loc[x.index, 'n'])
    )
    
    # Add level columns
    level_pivot = acc_all.pivot_table(
        index='model',
        columns='level',
        values='acc', 
        aggfunc=lambda x: np.average(x, weights=acc_all.loc[x.index, 'n'])
    )
    
    # Rename columns for clarity
    pivot_table.columns = ['Soccer', 'Volleyball', 'Basketball']
    level_pivot.columns = ['Level 0', 'Level 1', 'Level 2']
    
    # Combine the tables
    summary_table = pd.concat([pivot_table, level_pivot], axis=1)
    
    # Add overall accuracy column
    overall_acc = acc_all.groupby('model').apply(
        lambda x: np.average(x['acc'], weights=x['n'])
    )
    summary_table['Overall'] = overall_acc
    
    # Convert to percentages and round
    summary_table = (summary_table * 100).round(3)
    
    print(summary_table)
    
    # Print confidence intervals for key comparisons
    print("\n" + "="*120)
    print("KEY COMPARISONS WITH 95% CONFIDENCE INTERVALS")
    print("="*120)
    
    # Human vs AI models overall
    human_data = overall_by_model[overall_by_model['model'] == 'human'].iloc[0]
    ai_models = overall_by_model[overall_by_model['model'] != 'human']
    
    print(f"\nHuman Overall Accuracy: {human_data['overall_accuracy_pct']:.2f}%")
    print(f"95% CI: [{human_data['lower_ci_95']:.2f}%, {human_data['upper_ci_95']:.2f}%]")
    
    print("\nAI Models Overall Accuracy:")
    for _, model_data in ai_models.iterrows():
        print(f"{model_data['model'].upper()}: {model_data['overall_accuracy_pct']:.2f}%")
        print(f"  95% CI: [{model_data['lower_ci_95']:.2f}%, {model_data['upper_ci_95']:.2f}%]")
    
    # Sport comparisons
    print("\nSport Difficulty (lower = harder):")
    for _, sport_data in overall_by_sport.iterrows():
        sport_name = {'soccer': 'Soccer', 'vball': 'Volleyball', 'bb': 'Basketball'}[sport_data['sport']]
        print(f"{sport_name}: {sport_data['overall_accuracy_pct']:.2f}%")
        print(f"  95% CI: [{sport_data['lower_ci_95']:.2f}%, {sport_data['upper_ci_95']:.2f}%]")
    
    # Human performance across sports with detailed confidence intervals
    print("\n" + "="*120)
    print("HUMAN PERFORMANCE ACROSS SPORTS - DETAILED ANALYSIS")
    print("="*120)
    
    human_sport_data = acc_all[acc_all['model'] == 'human']
    
    print("\nHuman Accuracy by Sport (with 95% Confidence Intervals):")
    print("-" * 80)
    
    for sport in ['soccer', 'vball', 'bb']:
        sport_data = human_sport_data[human_sport_data['sport'] == sport]
        if not sport_data.empty:
            # Calculate confidence interval for human performance on this sport
            accuracies = sport_data['acc'].values
            sample_sizes = sport_data['n'].values
            
            # For human data, we can calculate more precise intervals since we have raw data
            # Calculate pooled standard error across all human predictions for this sport
            total_correct = sum(accuracies * sample_sizes)
            total_predictions = sum(sample_sizes)
            
            if total_predictions > 0:
                p_hat = total_correct / total_predictions
                se = np.sqrt(p_hat * (1 - p_hat) / total_predictions)
                
                # 95% confidence interval
                z_score = norm.ppf(0.975)  # 95% confidence level
                margin_of_error = z_score * se
                
                lower_ci = max(0, p_hat - margin_of_error)
                upper_ci = min(1, p_hat + margin_of_error)
                
                sport_name = {'soccer': 'Soccer', 'vball': 'Volleyball', 'bb': 'Basketball'}[sport]
                print(f"{sport_name}:")
                print(f"  Accuracy: {p_hat*100:.2f}%")
                print(f"  Total Predictions: {total_predictions:,}")
                print(f"  Correct Predictions: {total_correct:,.0f}")
                print(f"  Standard Error: {se*100:.3f}%")
                print(f"  95% CI: [{lower_ci*100:.2f}%, {upper_ci*100:.2f}%]")
                print(f"  Margin of Error: ±{margin_of_error*100:.2f}%")
                print()
    
    # Statistical significance testing between human performance across sports
    print("\nStatistical Significance Testing (Human Performance Across Sports):")
    print("-" * 80)
    
    # Create pairs for comparison
    sport_pairs = [('soccer', 'vball'), ('soccer', 'bb'), ('vball', 'bb')]
    
    for sport1, sport2 in sport_pairs:
        sport1_data = human_sport_data[human_sport_data['sport'] == sport1]
        sport2_data = human_sport_data[human_sport_data['sport'] == sport2]
        
        if not sport1_data.empty and not sport2_data.empty:
            # Calculate proportions and sample sizes
            p1 = sport1_data['acc'].iloc[0]
            n1 = sport1_data['n'].iloc[0]
            p2 = sport2_data['acc'].iloc[0]
            n2 = sport2_data['n'].iloc[0]
            
            # Two-proportion z-test
            # Manual implementation since proportions_ztest is not available in scipy.stats
            
            # Calculate counts of successes
            count1 = int(p1 * n1)
            count2 = int(p2 * n2)
            
            # Calculate pooled proportion
            pooled_p = (count1 + count2) / (n1 + n2)
            
            # Calculate standard error of the difference
            se_diff = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
            
            # Calculate z-statistic
            z_stat = (p1 - p2) / se_diff
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            
            sport1_name = {'soccer': 'Soccer', 'vball': 'Volleyball', 'bb': 'Basketball'}[sport1]
            sport2_name = {'soccer': 'Soccer', 'vball': 'Volleyball', 'bb': 'Basketball'}[sport2]
            
            print(f"{sport1_name} vs {sport2_name}:")
            print(f"  {sport1_name}: {p1*100:.2f}% (n={n1:,})")
            print(f"  {sport2_name}: {p2*100:.2f}% (n={n2:,})")
            print(f"  Difference: {(p1-p2)*100:+.2f} percentage points")
            print(f"  Z-statistic: {z_stat:.3f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
            print()
    
    print("="*120)
    
    return overall_by_model, overall_by_sport, overall_by_level

if __name__ == '__main__':
    run_all_metrics()
