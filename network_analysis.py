"""
network_analysis.py
===================
Weeks 3-5: Network Centrality, Comparison with Traditional Multipliers,
and Shock Propagation Simulation.

Covers:
  Week 3: Graph construction + 4 centrality measures
  Week 4: Spearman correlations, heatmaps, discrepancy analysis, 2×2 classification
  Week 5: Hypothetical extraction, GFC + COVID case studies, cascade simulation

Author: Aaryaveer
Project: Network Centrality Analysis of Input-Output Tables
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = '/home/claude/week1/outputs'
OUTPUT_DIR = '/home/claude/week3_5/outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
YEARS = [2007, 2012, 2017, 2022, 2024]

# Government sector codes to EXCLUDE
GOV_CODES = ['GFGD', 'GFGN', 'GFE', 'GSLG', 'GSLE']

# Edge weight threshold (drop negligible linkages)
EDGE_THRESHOLD = 0.001

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load all technology matrices and metadata from Week 1 outputs."""
    with open(os.path.join(INPUT_DIR, 'sector_codes.json'), 'r') as f:
        codes_dict = json.load(f)
    with open(os.path.join(INPUT_DIR, 'sector_names.json'), 'r') as f:
        names_dict = json.load(f)
    
    data = {}
    for year in YEARS:
        A = np.load(os.path.join(INPUT_DIR, f'A_{year}.npy'))
        L = np.load(os.path.join(INPUT_DIR, f'L_{year}.npy'))
        codes = codes_dict[str(year)]
        names = names_dict[str(year)]
        
        # Filter out government sectors
        keep_idx = [i for i, c in enumerate(codes) if c not in GOV_CODES]
        A_priv = A[np.ix_(keep_idx, keep_idx)]
        L_priv = L[np.ix_(keep_idx, keep_idx)]
        codes_priv = [codes[i] for i in keep_idx]
        names_priv = [names[i] for i in keep_idx]
        
        data[year] = {
            'A': A_priv, 'L': L_priv,
            'codes': codes_priv, 'names': names_priv,
            'n': len(keep_idx)
        }
    
    print(f"Loaded {len(YEARS)} years, {data[YEARS[0]]['n']} private sectors each")
    return data


# ============================================================
# WEEK 3: NETWORK CONSTRUCTION & CENTRALITY
# ============================================================

def build_graph(A, names, threshold=EDGE_THRESHOLD):
    """
    Build a weighted directed graph from the technology matrix.
    Edge (i → j) with weight A[i,j] means sector i supplies inputs to sector j.
    """
    n = A.shape[0]
    G = nx.DiGraph()
    
    for i in range(n):
        G.add_node(i, name=names[i])
    
    for i in range(n):
        for j in range(n):
            if A[i, j] > threshold:
                G.add_edge(i, j, weight=A[i, j])
    
    return G


def compute_centrality_measures(A, G, names):
    """
    Compute four centrality measures + traditional multipliers.
    
    Returns a DataFrame with all measures for each sector.
    """
    n = A.shape[0]
    I = np.eye(n)
    L = np.linalg.inv(I - A)
    
    # --- Traditional Multipliers ---
    BL = L.sum(axis=0)       # backward linkage (column sums)
    FL = L.sum(axis=1)       # forward linkage (row sums)
    BL_norm = BL / BL.mean()
    FL_norm = FL / FL.mean()
    
    # --- 1. Eigenvector Centrality ---
    try:
        eig_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-8)
        eig_cent_vals = np.array([eig_cent.get(i, 0) for i in range(n)])
    except nx.PowerIterationFailedConvergence:
        # Fallback: compute from Perron vector directly
        eigenvalues, eigenvectors = np.linalg.eig(A)
        idx = np.argmax(np.abs(eigenvalues))
        perron = np.abs(eigenvectors[:, idx])
        eig_cent_vals = perron / perron.sum()
    
    # --- 2. PageRank ---
    pagerank = nx.pagerank(G, weight='weight', alpha=0.85, max_iter=500, tol=1e-8)
    pagerank_vals = np.array([pagerank.get(i, 0) for i in range(n)])
    
    # --- 3. Betweenness Centrality ---
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    betweenness_vals = np.array([betweenness.get(i, 0) for i in range(n)])
    
    # --- 4. Katz Centrality ---
    # Katz with α < 1/ρ(A). Use α = 0.9/ρ(A) for safety margin.
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    alpha_katz = 0.9 / rho
    try:
        katz = nx.katz_centrality(G, alpha=alpha_katz, weight='weight',
                                  max_iter=1000, tol=1e-8)
        katz_vals = np.array([katz.get(i, 0) for i in range(n)])
    except Exception:
        # Fallback: compute directly as row sums of (I - αA)^{-1}
        L_katz = np.linalg.inv(I - alpha_katz * A)
        katz_vals = L_katz.sum(axis=1)
        katz_vals = katz_vals / katz_vals.sum()  # normalize
    
    # --- Build results DataFrame ---
    df = pd.DataFrame({
        'Sector': names,
        'BL': BL, 'BL_norm': BL_norm,
        'FL': FL, 'FL_norm': FL_norm,
        'EigCent': eig_cent_vals,
        'PageRank': pagerank_vals,
        'Betweenness': betweenness_vals,
        'Katz': katz_vals,
    })
    
    # Add ranks for all measures
    for col in ['BL_norm', 'FL_norm', 'EigCent', 'PageRank', 'Betweenness', 'Katz']:
        df[f'{col}_rank'] = df[col].rank(ascending=False).astype(int)
    
    return df


def run_week3(data):
    """Execute Week 3: network construction + centrality for all years."""
    print(f"\n{'='*70}")
    print("WEEK 3: NETWORK CENTRALITY ANALYSIS")
    print(f"{'='*70}")
    
    results = {}
    graphs = {}
    
    for year in YEARS:
        d = data[year]
        A, names, n = d['A'], d['names'], d['n']
        
        print(f"\n--- {year} ({n} private sectors) ---")
        
        # Build graph
        G = build_graph(A, names)
        graphs[year] = G
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
              f"(density: {nx.density(G):.3f})")
        
        # Centrality
        df = compute_centrality_measures(A, G, names)
        results[year] = df
        
        # Print top 5 for each measure
        for measure, label in [('BL_norm', 'Backward Linkage'), 
                                ('PageRank', 'PageRank'),
                                ('Betweenness', 'Betweenness'),
                                ('Katz', 'Katz Centrality')]:
            top = df.nlargest(5, measure)
            print(f"\n  Top 5 by {label}:")
            for _, row in top.iterrows():
                short_name = row['Sector'][:40]
                print(f"    {int(row[f'{measure}_rank']):>2}. {short_name:<42} {row[measure]:.4f}")
        
        # Save
        df.to_csv(os.path.join(OUTPUT_DIR, f'centrality_{year}.csv'), index=False)
    
    return results, graphs


# ============================================================
# WEEK 4: COMPARISON & CORRELATION ANALYSIS
# ============================================================

def run_week4(data, results):
    """Execute Week 4: compare traditional multipliers vs network centrality."""
    print(f"\n{'='*70}")
    print("WEEK 4: COMPARISON ANALYSIS")
    print(f"{'='*70}")
    
    measures = ['BL_norm', 'FL_norm', 'EigCent', 'PageRank', 'Betweenness', 'Katz']
    measure_labels = ['Backward\nLinkage', 'Forward\nLinkage', 'Eigenvector\nCentrality',
                      'PageRank', 'Betweenness\nCentrality', 'Katz\nCentrality']
    
    all_corr_matrices = {}
    
    for year in YEARS:
        df = results[year]
        n_measures = len(measures)
        corr_matrix = np.zeros((n_measures, n_measures))
        pval_matrix = np.zeros((n_measures, n_measures))
        
        for i, m1 in enumerate(measures):
            for j, m2 in enumerate(measures):
                rho, pval = spearmanr(df[m1], df[m2])
                corr_matrix[i, j] = rho
                pval_matrix[i, j] = pval
        
        all_corr_matrices[year] = corr_matrix
        
        print(f"\n--- {year} Spearman Rank Correlations ---")
        print(f"  BL vs PageRank:    ρ = {corr_matrix[0, 3]:.3f}")
        print(f"  BL vs Betweenness: ρ = {corr_matrix[0, 4]:.3f}")
        print(f"  BL vs Katz:        ρ = {corr_matrix[0, 5]:.3f}")
        print(f"  FL vs EigCent:     ρ = {corr_matrix[1, 2]:.3f}")
        print(f"  FL vs PageRank:    ρ = {corr_matrix[1, 3]:.3f}")
    
    # --- Figure: Correlation Heatmaps (all years) ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        cm = all_corr_matrices[year]
        mask = np.zeros_like(cm, dtype=bool)
        np.fill_diagonal(mask, True)
        
        sns.heatmap(cm, xticklabels=measure_labels, yticklabels=measure_labels if idx == 0 else False,
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-0.5, vmax=1.0,
                    ax=ax, mask=mask, square=True, cbar=idx == 4,
                    annot_kws={'size': 9})
        ax.set_title(f'{year}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
    
    plt.suptitle('Spearman Rank Correlations: Traditional Multipliers vs Network Centrality\n'
                 '(Values close to 1 = similar rankings, low values = network measures capture different info)',
                 fontsize=13, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  Saved correlation_heatmaps.png")
    
    # --- Figure: BL vs PageRank Scatter (key comparison) ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        df = results[year]
        
        ax.scatter(df['BL_norm'], df['PageRank'], alpha=0.6, s=40, c='steelblue', edgecolors='navy', linewidths=0.5)
        
        # Label outliers (top 5 by PageRank or BL that diverge)
        df['BL_rank_temp'] = df['BL_norm'].rank(ascending=False)
        df['PR_rank_temp'] = df['PageRank'].rank(ascending=False)
        df['rank_diff'] = abs(df['BL_rank_temp'] - df['PR_rank_temp'])
        
        # Label sectors in top 10 by either measure
        label_mask = (df['BL_rank_temp'] <= 7) | (df['PR_rank_temp'] <= 7)
        for _, row in df[label_mask].iterrows():
            short = row['Sector'][:18]
            ax.annotate(short, (row['BL_norm'], row['PageRank']),
                       fontsize=5.5, alpha=0.8, rotation=15,
                       textcoords="offset points", xytext=(3, 3))
        
        rho_val = spearmanr(df['BL_norm'], df['PageRank'])[0]
        ax.set_title(f'{year}  (ρ = {rho_val:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalised Backward Linkage' if idx == 2 else '', fontsize=9)
        if idx == 0:
            ax.set_ylabel('PageRank', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Backward Linkage vs PageRank — Do They Agree?\n'
                 '(Sectors far from the diagonal are "hidden critical" or "traditionally overrated")',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'BL_vs_PageRank_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved BL_vs_PageRank_scatter.png")
    
    # --- Discrepancy Analysis ---
    print(f"\n--- DISCREPANCY ANALYSIS: Sectors Ranked Very Differently ---")
    
    discrepancy_all = {}
    for year in YEARS:
        df = results[year]
        df['BL_rank_temp'] = df['BL_norm'].rank(ascending=False).astype(int)
        df['PR_rank_temp'] = df['PageRank'].rank(ascending=False).astype(int)
        df['BW_rank_temp'] = df['Betweenness'].rank(ascending=False).astype(int)
        
        # Hidden critical: low BL rank but high network centrality
        df['BL_PR_diff'] = df['BL_rank_temp'] - df['PR_rank_temp']  # positive = higher PageRank than BL
        df['BL_BW_diff'] = df['BL_rank_temp'] - df['BW_rank_temp']
        
        hidden = df[df['BL_PR_diff'] > 15].nlargest(5, 'BL_PR_diff')
        overrated = df[df['BL_PR_diff'] < -15].nsmallest(5, 'BL_PR_diff')
        
        print(f"\n  {year} — Hidden Critical (high PageRank, low BL):")
        for _, row in hidden.iterrows():
            print(f"    {row['Sector'][:40]:<42} BL_rank={row['BL_rank_temp']:>2} → PR_rank={row['PR_rank_temp']:>2}")
        
        print(f"  {year} — Traditionally Overrated (high BL, low PageRank):")
        for _, row in overrated.iterrows():
            print(f"    {row['Sector'][:40]:<42} BL_rank={row['BL_rank_temp']:>2} → PR_rank={row['PR_rank_temp']:>2}")
        
        discrepancy_all[year] = df[['Sector', 'BL_norm', 'PageRank', 'Betweenness',
                                     'BL_rank_temp', 'PR_rank_temp', 'BW_rank_temp',
                                     'BL_PR_diff', 'BL_BW_diff']]
        discrepancy_all[year].to_csv(os.path.join(OUTPUT_DIR, f'discrepancy_{year}.csv'), index=False)
    
    # --- 2×2 Classification Matrix ---
    print(f"\n--- 2×2 CLASSIFICATION (using 2024 data) ---")
    df = results[2024].copy()
    bl_median = df['BL_norm'].median()
    pr_median = df['PageRank'].median()
    
    df['Category'] = 'Other'
    df.loc[(df['BL_norm'] > bl_median) & (df['PageRank'] > pr_median), 'Category'] = 'Universally Critical'
    df.loc[(df['BL_norm'] <= bl_median) & (df['PageRank'] > pr_median), 'Category'] = 'Hidden Critical'
    df.loc[(df['BL_norm'] > bl_median) & (df['PageRank'] <= pr_median), 'Category'] = 'Traditionally Overrated'
    df.loc[(df['BL_norm'] <= bl_median) & (df['PageRank'] <= pr_median), 'Category'] = 'Not Critical'
    
    for cat in ['Universally Critical', 'Hidden Critical', 'Traditionally Overrated', 'Not Critical']:
        sectors = df[df['Category'] == cat]['Sector'].tolist()
        print(f"\n  {cat} ({len(sectors)} sectors):")
        for s in sectors[:8]:
            print(f"    • {s[:50]}")
        if len(sectors) > 8:
            print(f"    ... and {len(sectors)-8} more")
    
    # --- Figure: 2×2 Classification Scatter ---
    fig, ax = plt.subplots(figsize=(12, 9))
    cat_colors = {
        'Universally Critical': '#d62728',
        'Hidden Critical': '#ff7f0e',
        'Traditionally Overrated': '#9467bd',
        'Not Critical': '#aec7e8'
    }
    
    for cat, color in cat_colors.items():
        mask = df['Category'] == cat
        ax.scatter(df.loc[mask, 'BL_norm'], df.loc[mask, 'PageRank'],
                  c=color, label=cat, s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Label all universally critical and hidden critical
    for _, row in df[df['Category'].isin(['Universally Critical', 'Hidden Critical'])].iterrows():
        ax.annotate(row['Sector'][:22], (row['BL_norm'], row['PageRank']),
                   fontsize=6.5, alpha=0.85, textcoords="offset points", xytext=(4, 4))
    
    ax.axvline(x=bl_median, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=pr_median, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalised Backward Linkage (Traditional)', fontsize=12)
    ax.set_ylabel('PageRank (Network)', fontsize=12)
    ax.set_title('2×2 Classification: Traditional vs Network Importance — 2024\n'
                 'Hidden Critical sectors are missed by traditional Leontief analysis',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'classification_2x2_2024.png'), dpi=300)
    plt.close()
    print("\n  Saved classification_2x2_2024.png")
    
    # --- Correlation trend over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    pairs = [
        ('BL_norm', 'PageRank', 'BL vs PageRank'),
        ('BL_norm', 'Betweenness', 'BL vs Betweenness'),
        ('BL_norm', 'Katz', 'BL vs Katz'),
        ('FL_norm', 'EigCent', 'FL vs Eigenvector'),
        ('FL_norm', 'PageRank', 'FL vs PageRank'),
    ]
    
    for m1, m2, label in pairs:
        corrs = []
        for year in YEARS:
            rho_val, _ = spearmanr(results[year][m1], results[year][m2])
            corrs.append(rho_val)
        ax.plot(YEARS, corrs, 'o-', linewidth=2, markersize=8, label=label)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Spearman Rank Correlation', fontsize=12)
    ax.set_title('Do Traditional and Network Measures Diverge Over Time?\n'
                 '(Decreasing correlation = network measures reveal more hidden structure)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)
    
    # Add GFC and COVID markers
    ax.axvspan(2007, 2009, alpha=0.1, color='red', label='GFC')
    ax.axvspan(2020, 2021, alpha=0.1, color='orange', label='COVID')
    ax.annotate('GFC', (2008, ax.get_ylim()[1]*0.95), fontsize=9, color='red', alpha=0.7)
    ax.annotate('COVID', (2020.5, ax.get_ylim()[1]*0.95), fontsize=9, color='orange', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_trend_over_time.png'), dpi=300)
    plt.close()
    print("  Saved correlation_trend_over_time.png")
    
    return all_corr_matrices, discrepancy_all


# ============================================================
# WEEK 5: SHOCK PROPAGATION SIMULATION
# ============================================================

def hypothetical_extraction(A, x, sector_idx):
    """
    Hypothetical Extraction Method: remove sector j and compute GDP loss.
    
    1. Set row j and column j of A to zero (sector j ceases to exist)
    2. Recompute the Leontief inverse for the reduced system
    3. GDP loss = difference in total output
    
    This is the standard method in IO analysis for measuring
    sector criticality — analogous to stress testing in finance.
    """
    n = A.shape[0]
    A_reduced = A.copy()
    A_reduced[sector_idx, :] = 0.0  # Sector stops supplying
    A_reduced[:, sector_idx] = 0.0  # Sector stops demanding
    
    I = np.eye(n)
    
    # Original total output
    L_original = np.linalg.inv(I - A)
    
    # Reduced total output
    L_reduced = np.linalg.inv(I - A_reduced)
    
    # Impact = difference in total requirements
    # Use column sums as proxy for total output multiplier
    original_total = L_original.sum()
    reduced_total = L_reduced.sum()
    
    impact = original_total - reduced_total
    impact_pct = impact / original_total * 100
    
    return impact, impact_pct


def cascade_failure(A, initial_sector, threshold=0.15):
    """
    Cascading failure simulation:
    1. Remove the initial sector
    2. Check which remaining sectors lose more than `threshold` fraction
       of their inputs → they also fail
    3. Repeat until no more sectors fail
    
    Returns the set of failed sectors and number of cascade rounds.
    """
    n = A.shape[0]
    failed = {initial_sector}
    active = set(range(n)) - failed
    rounds = 0
    
    while True:
        rounds += 1
        new_failures = set()
        
        for j in active:
            # Fraction of sector j's inputs coming from failed sectors
            total_input = A[:, j].sum()
            if total_input > 0:
                failed_input = sum(A[i, j] for i in failed)
                loss_frac = failed_input / total_input
                if loss_frac > threshold:
                    new_failures.add(j)
        
        if not new_failures:
            break
        
        failed.update(new_failures)
        active -= new_failures
        
        if rounds > n:  # Safety
            break
    
    return failed, rounds


def run_week5(data, results):
    """Execute Week 5: shock propagation and case studies."""
    print(f"\n{'='*70}")
    print("WEEK 5: SHOCK PROPAGATION SIMULATION")
    print(f"{'='*70}")
    
    extraction_results = {}
    cascade_results = {}
    
    for year in YEARS:
        d = data[year]
        A, names, n = d['A'], d['names'], d['n']
        x = np.ones(n)  # normalized
        
        print(f"\n--- {year}: Hypothetical Extraction (all {n} sectors) ---")
        
        impacts = []
        for j in range(n):
            impact, impact_pct = hypothetical_extraction(A, x, j)
            impacts.append({
                'Sector': names[j],
                'Extraction_Impact': impact,
                'Impact_Pct': impact_pct,
            })
        
        df_impact = pd.DataFrame(impacts)
        df_impact['Impact_Rank'] = df_impact['Impact_Pct'].rank(ascending=False).astype(int)
        
        # Merge with centrality measures
        df_cent = results[year].copy()
        df_merged = df_impact.merge(df_cent, on='Sector')
        extraction_results[year] = df_merged
        
        # Print top 10 by extraction impact
        top10 = df_merged.nlargest(10, 'Impact_Pct')
        print(f"\n  Top 10 Most Critical (by extraction impact):")
        for _, row in top10.iterrows():
            print(f"    {row['Impact_Rank']:>2}. {row['Sector'][:40]:<42} "
                  f"Impact={row['Impact_Pct']:.2f}%  BL_rank={int(row['BL_norm_rank']):>2}  "
                  f"PR_rank={int(row['PageRank_rank']):>2}")
        
        # Compute R² for each centrality measure vs extraction impact
        print(f"\n  Predictive Power (Spearman ρ with extraction impact):")
        predictors = ['BL_norm', 'FL_norm', 'EigCent', 'PageRank', 'Betweenness', 'Katz']
        pred_labels = ['Backward Linkage', 'Forward Linkage', 'Eigenvector Cent.',
                       'PageRank', 'Betweenness Cent.', 'Katz Centrality']
        
        for pred, label in zip(predictors, pred_labels):
            rho_val, pval = spearmanr(df_merged[pred], df_merged['Impact_Pct'])
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"    {label:<22} ρ = {rho_val:.4f} {sig}")
        
        # Cascade failure simulation
        print(f"\n  Cascade Failure Simulation (threshold=15% input loss):")
        cascades = []
        for j in range(n):
            failed_set, rounds = cascade_failure(A, j, threshold=0.15)
            cascades.append({
                'Sector': names[j],
                'Cascade_Size': len(failed_set),
                'Cascade_Rounds': rounds,
            })
        
        df_cascade = pd.DataFrame(cascades)
        df_cascade['Cascade_Rank'] = df_cascade['Cascade_Size'].rank(ascending=False, method='min').astype(int)
        cascade_results[year] = df_cascade
        
        top5_cascade = df_cascade.nlargest(5, 'Cascade_Size')
        for _, row in top5_cascade.iterrows():
            print(f"    {row['Sector'][:40]:<42} → {row['Cascade_Size']} sectors fail "
                  f"in {row['Cascade_Rounds']} rounds")
        
        # Save
        df_merged.to_csv(os.path.join(OUTPUT_DIR, f'extraction_{year}.csv'), index=False)
        df_cascade.to_csv(os.path.join(OUTPUT_DIR, f'cascade_{year}.csv'), index=False)
    
    # ================================================================
    # CASE STUDY 1: GFC (2007 → 2012)
    # ================================================================
    print(f"\n{'='*70}")
    print("CASE STUDY 1: GLOBAL FINANCIAL CRISIS (2007 → 2012)")
    print(f"{'='*70}")
    
    df_07 = extraction_results[2007]
    df_12 = extraction_results[2012]
    
    merged_gfc = df_07[['Sector', 'Impact_Pct', 'PageRank', 'Betweenness']].merge(
        df_12[['Sector', 'Impact_Pct', 'PageRank', 'Betweenness']],
        on='Sector', suffixes=('_2007', '_2012'))
    
    merged_gfc['Impact_Change'] = merged_gfc['Impact_Pct_2012'] - merged_gfc['Impact_Pct_2007']
    merged_gfc['PR_Change'] = merged_gfc['PageRank_2012'] - merged_gfc['PageRank_2007']
    
    print("\n  Sectors with LARGEST increase in criticality (2007→2012):")
    top_increase = merged_gfc.nlargest(8, 'Impact_Change')
    for _, row in top_increase.iterrows():
        print(f"    {row['Sector'][:40]:<42} Impact: {row['Impact_Pct_2007']:.2f}% → {row['Impact_Pct_2012']:.2f}% "
              f"(+{row['Impact_Change']:.2f}%)")
    
    print("\n  Sectors with LARGEST decrease in criticality (2007→2012):")
    top_decrease = merged_gfc.nsmallest(8, 'Impact_Change')
    for _, row in top_decrease.iterrows():
        print(f"    {row['Sector'][:40]:<42} Impact: {row['Impact_Pct_2007']:.2f}% → {row['Impact_Pct_2012']:.2f}% "
              f"({row['Impact_Change']:.2f}%)")
    
    merged_gfc.to_csv(os.path.join(OUTPUT_DIR, 'case_study_gfc.csv'), index=False)
    
    # ================================================================
    # CASE STUDY 2: COVID-19 (2017 → 2022)
    # ================================================================
    print(f"\n{'='*70}")
    print("CASE STUDY 2: COVID-19 PANDEMIC (2017 → 2022)")
    print(f"{'='*70}")
    
    df_17 = extraction_results[2017]
    df_22 = extraction_results[2022]
    
    merged_covid = df_17[['Sector', 'Impact_Pct', 'PageRank', 'Betweenness']].merge(
        df_22[['Sector', 'Impact_Pct', 'PageRank', 'Betweenness']],
        on='Sector', suffixes=('_2017', '_2022'))
    
    merged_covid['Impact_Change'] = merged_covid['Impact_Pct_2022'] - merged_covid['Impact_Pct_2017']
    merged_covid['PR_Change'] = merged_covid['PageRank_2022'] - merged_covid['PageRank_2017']
    
    print("\n  Sectors with LARGEST increase in criticality (2017→2022):")
    top_increase = merged_covid.nlargest(8, 'Impact_Change')
    for _, row in top_increase.iterrows():
        print(f"    {row['Sector'][:40]:<42} Impact: {row['Impact_Pct_2017']:.2f}% → {row['Impact_Pct_2022']:.2f}% "
              f"(+{row['Impact_Change']:.2f}%)")
    
    print("\n  Sectors with LARGEST decrease in criticality (2017→2022):")
    top_decrease = merged_covid.nsmallest(8, 'Impact_Change')
    for _, row in top_decrease.iterrows():
        print(f"    {row['Sector'][:40]:<42} Impact: {row['Impact_Pct_2017']:.2f}% → {row['Impact_Pct_2022']:.2f}% "
              f"({row['Impact_Change']:.2f}%)")
    
    merged_covid.to_csv(os.path.join(OUTPUT_DIR, 'case_study_covid.csv'), index=False)
    
    # ================================================================
    # KEY FIGURES
    # ================================================================
    
    # --- Figure: Predictive Power Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    predictors = ['BL_norm', 'FL_norm', 'EigCent', 'PageRank', 'Betweenness', 'Katz']
    pred_labels = ['Backward\nLinkage', 'Forward\nLinkage', 'Eigenvector\nCent.',
                   'PageRank', 'Betweenness\nCent.', 'Katz\nCent.']
    
    x_pos = np.arange(len(predictors))
    width = 0.15
    
    colors_year = {2007: '#1f77b4', 2012: '#ff7f0e', 2017: '#2ca02c', 
                   2022: '#d62728', 2024: '#9467bd'}
    
    for i, year in enumerate(YEARS):
        df = extraction_results[year]
        rhos_year = []
        for pred in predictors:
            rho_val, _ = spearmanr(df[pred], df['Impact_Pct'])
            rhos_year.append(rho_val)
        
        ax.bar(x_pos + i * width, rhos_year, width, label=str(year),
               color=colors_year[year], alpha=0.85)
    
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(pred_labels, fontsize=10)
    ax.set_ylabel('Spearman ρ with Extraction Impact', fontsize=12)
    ax.set_title('Which Measure Best Predicts Sector Criticality?\n'
                 '(Higher ρ = better at identifying systemically important sectors)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictive_power_comparison.png'), dpi=300)
    plt.close()
    print("\n  Saved predictive_power_comparison.png")
    
    # --- Figure: GFC Impact Change ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for idx, (merged, title_event, years_pair) in enumerate([
        (merged_gfc, 'Global Financial Crisis', '2007 → 2012'),
        (merged_covid, 'COVID-19 Pandemic', '2017 → 2022')
    ]):
        ax = axes[idx]
        sorted_df = merged.sort_values('Impact_Change')
        
        # Show top 10 increases and top 10 decreases
        show = pd.concat([sorted_df.head(10), sorted_df.tail(10)])
        
        colors_bar = ['#d62728' if v > 0 else '#2ca02c' for v in show['Impact_Change']]
        
        y_pos = range(len(show))
        ax.barh(y_pos, show['Impact_Change'].values, color=colors_bar, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s[:30] for s in show['Sector'].values], fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Change in Extraction Impact (%)', fontsize=11)
        ax.set_title(f'{title_event} ({years_pair})\n'
                     f'Red = became more critical, Green = became less critical',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'case_studies_impact_change.png'), dpi=300)
    plt.close()
    print("  Saved case_studies_impact_change.png")
    
    # --- Figure: Extraction Impact vs PageRank (headline result) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    key_years = [2007, 2017, 2024]
    
    for idx, year in enumerate(key_years):
        ax = axes[idx]
        df = extraction_results[year]
        
        ax.scatter(df['PageRank'], df['Impact_Pct'], alpha=0.6, s=45,
                  c='steelblue', edgecolors='navy', linewidths=0.5)
        
        # Fit line
        z = np.polyfit(df['PageRank'], df['Impact_Pct'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['PageRank'].min(), df['PageRank'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, alpha=0.7)
        
        rho_val, _ = spearmanr(df['PageRank'], df['Impact_Pct'])
        
        # Label top 5 critical
        for _, row in df.nlargest(5, 'Impact_Pct').iterrows():
            ax.annotate(row['Sector'][:20], (row['PageRank'], row['Impact_Pct']),
                       fontsize=6.5, textcoords="offset points", xytext=(4, 4))
        
        ax.set_xlabel('PageRank', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Extraction Impact (%)', fontsize=11)
        ax.set_title(f'{year}  ρ = {rho_val:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('PageRank vs Actual Sector Criticality (Extraction Impact)\n'
                 'Higher correlation = PageRank accurately predicts systemic importance',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pagerank_vs_extraction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved pagerank_vs_extraction.png")
    
    # --- Figure: Cascade Size Distribution ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        df_c = cascade_results[year]
        sizes = df_c['Cascade_Size'].values
        
        ax.hist(sizes, bins=range(1, max(sizes)+2), color='steelblue', 
                edgecolor='navy', alpha=0.8)
        ax.set_xlabel('Cascade Size (# sectors failed)' if idx == 2 else '', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{year}  max={max(sizes)}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Distribution of Cascade Failure Sizes\n'
                 '(How many sectors fail when each individual sector is removed)',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cascade_size_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved cascade_size_distribution.png")
    
    return extraction_results, cascade_results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NETWORK CENTRALITY ANALYSIS OF U.S. INPUT-OUTPUT TABLES")
    print("Weeks 3-5: Centrality, Comparison, Shock Simulation")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Week 3: Centrality
    results, graphs = run_week3(data)
    
    # Week 4: Comparison
    corr_matrices, discrepancies = run_week4(data, results)
    
    # Week 5: Shock simulation + case studies
    extraction_results, cascade_results = run_week5(data, results)
    
    # ================================================================
    # FINAL SUMMARY — KEY RESULTS FOR THE PAPER
    # ================================================================
    print(f"\n{'='*70}")
    print("KEY RESULTS FOR THE PAPER")
    print(f"{'='*70}")
    
    print("\n1. PREDICTIVE POWER RANKING (2024, Spearman ρ with extraction impact):")
    df = extraction_results[2024]
    predictors = ['BL_norm', 'FL_norm', 'EigCent', 'PageRank', 'Betweenness', 'Katz']
    pred_labels = ['Backward Linkage', 'Forward Linkage', 'Eigenvector Cent.',
                   'PageRank', 'Betweenness Cent.', 'Katz Centrality']
    pred_scores = []
    for pred, label in zip(predictors, pred_labels):
        rho_val, pval = spearmanr(df[pred], df['Impact_Pct'])
        pred_scores.append((label, rho_val, pval))
    
    pred_scores.sort(key=lambda x: -x[1])
    for label, rho_val, pval in pred_scores:
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        marker = " ← HEADLINE" if label == 'PageRank' or rho_val == max(s[1] for s in pred_scores) else ""
        print(f"  {label:<22} ρ = {rho_val:.4f} ({sig}){marker}")
    
    print(f"\n2. SPECTRAL RADIUS TREND:")
    for year in YEARS:
        rho = np.max(np.abs(np.linalg.eigvals(data[year]['A'])))
        print(f"  {year}: ρ = {rho:.6f}")
    
    print(f"\n3. FILES GENERATED:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not os.path.isdir(os.path.join(OUTPUT_DIR, f)):
            print(f"  {f}")
    print(f"\n  Figures:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f"    {f}")
    
    print(f"\n{'='*70}")
    print("WEEKS 3-5 COMPLETE!")
    print(f"{'='*70}")
    print(f"Ready for Week 6 (spectral analysis) and Week 7 (paper writing)!")


if __name__ == '__main__':
    main()
