"""
bea_parser.py
=============
Parses BEA Use Tables (Before Redefinitions, Producers' Prices, Summary level)
into technology matrices for the Leontief input-output model.

Tuned to the exact Excel format from BEA's Interactive Data Tables.

File structure:
  Row 0:  Title
  Row 1:  Year and units (Millions of dollars)
  Row 2:  Bureau of Economic Analysis
  Rows 3-4: blank
  Row 5:  Industry codes header (cols 2-72 = 71 industries)
  Row 6:  Industry names header
  Rows 7-77:  Commodity × Industry intermediate flows (Z matrix)
  Row 78: Used/Scrap
  Row 79: Other/Noncomparable imports
  Row 80: Total Intermediate
  Row 81: V001 — Compensation of employees
  Row 82: V002 — Taxes on production
  Row 83: V003 — Gross operating surplus
  Row 84: Total Value Added
  Row 85: Total Industry Output  ← x vector

Columns:
  Col 0:  Commodity codes
  Col 1:  Commodity names
  Cols 2-72: Industry columns (71 industries)
  Col 73: Total Intermediate (by commodity)
  Cols 74+: Final demand categories

Author: Aaryaveer
Project: Network Centrality Analysis of Input-Output Tables
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    2007: '/mnt/user-data/uploads/Table_2007.xlsx',
    2012: '/mnt/user-data/uploads/Table_2012.xlsx',
    2017: '/mnt/user-data/uploads/Table_2017.xlsx',
    2022: '/mnt/user-data/uploads/Table_2022.xlsx',
    2024: '/mnt/user-data/uploads/Table_2024.xlsx',
}

OUTPUT_DIR = '/home/claude/week1/outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# BEA table layout constants
HEADER_ROW = 5       # Row with industry codes
NAME_ROW = 6         # Row with industry names
DATA_START_ROW = 7   # First commodity data row
CODE_COL = 0         # Column with commodity codes
NAME_COL = 1         # Column with commodity names
INDUSTRY_START_COL = 2  # First industry column

# ============================================================
# PARSER
# ============================================================

def to_numeric(val):
    """Convert BEA cell values to float. Handles '---', '(D)', negatives in parens."""
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return 0.0
        return float(val)
    s = str(val).strip().replace(',', '')
    if s in ['---', '(D)', '(S)', 'nan', 'None', '', '-', '*', '...']:
        return 0.0
    if s.startswith('(') and s.endswith(')'):
        return -float(s[1:-1])
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_bea_file(filepath, year):
    """
    Parse a single BEA Use Table Excel file.
    
    Returns:
        Z: intermediate use matrix (n × n) in millions of dollars
        x: total industry output vector (n,) in millions of dollars
        sector_codes: list of BEA industry codes
        sector_names: list of industry names
    """
    print(f"\n{'='*60}")
    print(f"Parsing {year}: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    df = pd.read_excel(filepath, header=None, sheet_name=0)
    print(f"  Raw shape: {df.shape}")
    
    # --- Identify industry columns ---
    # Row 5 has industry codes starting from col 2
    # Find where industry columns end (look for 'nan' or final demand codes starting with 'F')
    industry_codes = []
    industry_cols = []
    industry_names = []
    
    for j in range(INDUSTRY_START_COL, df.shape[1]):
        code = str(df.iloc[HEADER_ROW, j]).strip()
        if code == 'nan' or code.startswith('F') or code.startswith('T'):
            break
        industry_codes.append(code)
        industry_cols.append(j)
        name = str(df.iloc[NAME_ROW, j]).strip()
        industry_names.append(name)
    
    n_industries = len(industry_codes)
    print(f"  Found {n_industries} industry columns")
    
    # --- Identify commodity rows ---
    commodity_codes = []
    commodity_rows = []
    commodity_names = []
    
    for i in range(DATA_START_ROW, df.shape[0]):
        code = str(df.iloc[i, CODE_COL]).strip()
        # Stop when we hit non-industry codes (Used, Other, V001, etc.)
        if code in ['Used', 'Other', 'nan', 'V001', 'V002', 'V003', ''] or code.startswith('Legend'):
            break
        commodity_codes.append(code)
        commodity_rows.append(i)
        name = str(df.iloc[i, NAME_COL]).strip()
        commodity_names.append(name)
    
    n_commodities = len(commodity_codes)
    print(f"  Found {n_commodities} commodity rows")
    
    # --- Verify codes match ---
    if set(commodity_codes) != set(industry_codes):
        missing_in_cols = set(commodity_codes) - set(industry_codes)
        missing_in_rows = set(industry_codes) - set(commodity_codes)
        if missing_in_cols:
            print(f"  ⚠️  Codes in rows but not columns: {missing_in_cols}")
        if missing_in_rows:
            print(f"  ⚠️  Codes in columns but not rows: {missing_in_rows}")
    
    # Use intersection for square matrix
    common_codes = [c for c in commodity_codes if c in industry_codes]
    n = len(common_codes)
    print(f"  Square matrix size: {n} × {n}")
    
    # Build index mappings
    code_to_row = {code: row for code, row in zip(commodity_codes, commodity_rows)}
    code_to_col = {code: col for code, col in zip(industry_codes, industry_cols)}
    code_to_name = {code: name for code, name in zip(commodity_codes, commodity_names)}
    # Also get names from industry side (sometimes more complete)
    for code, name in zip(industry_codes, industry_names):
        if code not in code_to_name:
            code_to_name[code] = name
    
    # --- Extract Z matrix ---
    Z = np.zeros((n, n))
    for i, code_i in enumerate(common_codes):
        for j, code_j in enumerate(common_codes):
            row_idx = code_to_row[code_i]
            col_idx = code_to_col[code_j]
            Z[i, j] = to_numeric(df.iloc[row_idx, col_idx])
    
    # --- Extract total industry output ---
    # Find the "Total Industry Output" row
    x = np.zeros(n)
    output_row = None
    
    for i in range(DATA_START_ROW, df.shape[0]):
        label = str(df.iloc[i, NAME_COL]).strip().lower()
        if 'total industry output' in label:
            output_row = i
            break
    
    if output_row is not None:
        print(f"  Found 'Total Industry Output' at row {output_row}")
        for j, code_j in enumerate(common_codes):
            col_idx = code_to_col[code_j]
            x[j] = to_numeric(df.iloc[output_row, col_idx])
    else:
        # Fallback: Total Intermediate + Total Value Added
        print("  ⚠️  'Total Industry Output' row not found, computing from components")
        ti_row = None
        va_row = None
        for i in range(DATA_START_ROW, df.shape[0]):
            label = str(df.iloc[i, NAME_COL]).strip().lower()
            if 'total intermediate' in label:
                ti_row = i
            if 'total value added' in label:
                va_row = i
        
        if ti_row and va_row:
            for j, code_j in enumerate(common_codes):
                col_idx = code_to_col[code_j]
                ti = to_numeric(df.iloc[ti_row, col_idx])
                va = to_numeric(df.iloc[va_row, col_idx])
                x[j] = ti + va
    
    sector_names = [code_to_name.get(c, c) for c in common_codes]
    
    # --- Print summary ---
    print(f"\n  Z matrix: {Z.shape}, sum = ${Z.sum()/1e6:.1f} trillion")
    print(f"  Total output: sum = ${x.sum()/1e6:.1f} trillion")
    print(f"  Non-zero entries in Z: {np.sum(Z != 0)} / {n*n} ({np.sum(Z != 0)/(n*n):.1%})")
    
    # Check for any zero-output sectors
    zero_output = [common_codes[j] for j in range(n) if x[j] <= 0]
    if zero_output:
        print(f"  ⚠️  Zero-output sectors: {zero_output}")
    
    return Z, x, common_codes, sector_names


def construct_technology_matrix(Z, x):
    """
    A[i,j] = Z[i,j] / x[j]
    
    To produce $1 of output in sector j, you need $A[i,j] of inputs from sector i.
    """
    n = Z.shape[0]
    A = np.zeros((n, n))
    for j in range(n):
        if x[j] > 0:
            A[:, j] = Z[:, j] / x[j]
    return A


def validate(A, year, sector_names):
    """Run all sanity checks and return stats dict."""
    n = A.shape[0]
    print(f"\n  --- Validation ---")
    
    # Non-negativity (small negatives from BEA adjustments are OK)
    neg_count = np.sum(A < -1e-10)
    if neg_count > 0:
        print(f"  ⚠️  {neg_count} negative entries (min={A.min():.6f}) → clipping to 0")
        A[A < 0] = 0.0
    else:
        print(f"  ✓ All entries non-negative")
    
    # Column sums
    col_sums = A.sum(axis=0)
    over = np.sum(col_sums >= 1.0)
    print(f"  Column sums: min={col_sums.min():.4f}, max={col_sums.max():.4f}, mean={col_sums.mean():.4f}")
    if over > 0:
        bad = [(sector_names[j], col_sums[j]) for j in range(n) if col_sums[j] >= 1.0]
        print(f"  ⚠️  {over} columns with sum ≥ 1: {bad[:3]}")
    else:
        print(f"  ✓ All column sums < 1")
    
    # Spectral radius
    eigenvalues = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigenvalues))
    print(f"  Spectral radius ρ(A) = {rho:.6f}")
    if rho < 1.0:
        terms = int(np.ceil(np.log(1e-6) / np.log(rho)))
        print(f"  ✓ ρ(A) < 1 → Neumann series converges (~{terms} terms for 10⁻⁶ accuracy)")
    else:
        print(f"  ⚠️  ρ(A) ≥ 1 → Neumann series does NOT converge!")
    
    # Matrix density
    density = np.sum(np.abs(A) > 1e-10) / (n * n)
    print(f"  Density: {density:.1%}")
    
    return {
        'year': year, 'n': n, 'rho': rho,
        'col_sum_min': col_sums.min(), 'col_sum_max': col_sums.max(),
        'col_sum_mean': col_sums.mean(), 'density': density,
    }


def neumann_convergence(A, max_k=100, tol=1e-10):
    """Compute Neumann series and track convergence."""
    n = A.shape[0]
    I = np.eye(n)
    L_exact = np.linalg.inv(I - A)
    
    L_k = I.copy()
    A_power = I.copy()
    errors = []
    
    for k in range(1, max_k + 1):
        A_power = A_power @ A
        L_k = L_k + A_power
        err = np.linalg.norm(L_exact - L_k, 'fro')
        errors.append({'k': k, 'error': err, 'power_norm': np.linalg.norm(A_power, 'fro')})
        if err < tol:
            return L_exact, errors, k
    
    return L_exact, errors, max_k


def compute_multipliers(A, L, sector_names):
    """Compute backward/forward linkages."""
    BL = L.sum(axis=0)
    FL = L.sum(axis=1)
    BL_norm = BL / BL.mean()
    FL_norm = FL / FL.mean()
    
    df = pd.DataFrame({
        'Sector': sector_names,
        'BL': BL, 'BL_norm': BL_norm, 'BL_rank': pd.Series(BL_norm).rank(ascending=False).astype(int),
        'FL': FL, 'FL_norm': FL_norm, 'FL_rank': pd.Series(FL_norm).rank(ascending=False).astype(int),
    })
    return df.sort_values('BL_norm', ascending=False)


# ============================================================
# FIGURES
# ============================================================

def generate_all_figures(matrices, sector_names_dict, all_errors):
    """Generate all Week 1 publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # --- 1. Heatmaps ---
    for year, A in sorted(matrices.items()):
        fig, ax = plt.subplots(figsize=(16, 14))
        A_display = A.copy()
        A_display[A_display == 0] = np.nan
        
        names = [n[:25] for n in sector_names_dict[year]]
        sns.heatmap(np.log10(A_display + 1e-10), xticklabels=names, yticklabels=names,
                    cmap='YlOrRd', ax=ax, cbar_kws={'label': 'log₁₀(A[i,j])'})
        ax.set_title(f'Technology Matrix A — {year}\n(log scale, darker = stronger input requirement)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Purchasing Industry (j)', fontsize=11)
        ax.set_ylabel('Supplying Commodity (i)', fontsize=11)
        ax.tick_params(axis='both', labelsize=5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_A_{year}.png'), dpi=300)
        plt.close()
    print("  Saved heatmaps")
    
    # --- 2. Neumann Series Convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'2007': '#1f77b4', '2012': '#ff7f0e', '2017': '#2ca02c', 
              '2022': '#d62728', '2024': '#9467bd'}
    
    for year, errors in sorted(all_errors.items()):
        ks = [e['k'] for e in errors]
        errs = [e['error'] for e in errors]
        norms = [e['power_norm'] for e in errors]
        c = colors.get(str(year), 'gray')
        axes[0].semilogy(ks, errs, linewidth=2, label=str(year), color=c)
        axes[1].semilogy(ks[:30], norms[:30], 'o-', linewidth=1.5, markersize=3, 
                         label=str(year), color=c)
    
    axes[0].set_xlabel('Terms in Neumann Series (k)', fontsize=12)
    axes[0].set_ylabel('‖(I−A)⁻¹ − Σ Aᵗ‖_F', fontsize=12)
    axes[0].set_title('Convergence of Neumann Series', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1e-6, color='red', linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Power k', fontsize=12)
    axes[1].set_ylabel('‖Aᵏ‖_F', fontsize=12)
    axes[1].set_title('Geometric Decay of Matrix Powers', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'neumann_convergence.png'), dpi=300)
    plt.close()
    print("  Saved neumann_convergence.png")
    
    # --- 3. Spectral Radius Over Time ---
    years = sorted(matrices.keys())
    rhos = [np.max(np.abs(np.linalg.eigvals(matrices[y]))) for y in years]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, rhos, 'bo-', linewidth=2.5, markersize=12, markerfacecolor='steelblue')
    for yr, rho in zip(years, rhos):
        ax.annotate(f'{rho:.4f}', (yr, rho), textcoords="offset points",
                    xytext=(0, 14), ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Spectral Radius ρ(A)', fontsize=12)
    ax.set_title('Spectral Radius of U.S. Technology Matrix Over Time\n'
                 '(higher ρ = longer supply chains = slower Neumann convergence)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(rhos) * 0.85, max(rhos) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'spectral_radius_over_time.png'), dpi=300)
    plt.close()
    print("  Saved spectral_radius_over_time.png")
    
    # --- 4. Eigenvalue Spectra ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    
    for idx, year in enumerate(sorted(matrices.keys())):
        if idx >= 5:
            break
        ax = axes_flat[idx]
        eigs = np.linalg.eigvals(matrices[year])
        rho = np.max(np.abs(eigs))
        
        ax.scatter(eigs.real, eigs.imag, c='steelblue', alpha=0.7, s=50, 
                   edgecolors='navy', linewidths=0.5)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.2, linewidth=1)
        ax.plot(rho*np.cos(theta), rho*np.sin(theta), 'g-', alpha=0.5, linewidth=1.5)
        ax.set_title(f'{year}  ρ = {rho:.4f}', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Re(λ)', fontsize=9)
        ax.set_ylabel('Im(λ)', fontsize=9)
    
    # Hide unused subplot
    if len(matrices) < 6:
        axes_flat[5].set_visible(False)
    
    plt.suptitle('Eigenvalue Spectra of U.S. Technology Matrices\n'
                 'Green circle = spectral radius, Red = unit circle',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eigenvalue_spectra.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved eigenvalue_spectra.png")
    
    # --- 5. Top 10 Sectors by Backward Linkage (2024) ---
    latest_year = max(matrices.keys())
    A = matrices[latest_year]
    L = np.linalg.inv(np.eye(A.shape[0]) - A)
    names = sector_names_dict[latest_year]
    BL = L.sum(axis=0)
    BL_norm = BL / BL.mean()
    
    top_idx = np.argsort(BL_norm)[-15:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    top_names = [names[i][:35] for i in top_idx]
    top_vals = BL_norm[top_idx]
    colors_bar = ['#d62728' if v > 1 else '#aec7e8' for v in top_vals]
    
    ax.barh(range(len(top_idx)), top_vals[::-1], color=colors_bar[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Mean = 1.0')
    ax.set_xlabel('Normalised Backward Linkage', fontsize=12)
    ax.set_title(f'Top 15 Sectors by Backward Linkage — {latest_year}\n'
                 f'(Red = above average demand-pull impact)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'top15_backward_linkage_{latest_year}.png'), dpi=300)
    plt.close()
    print(f"  Saved top15_backward_linkage_{latest_year}.png")
    
    # --- 6. Row Sum Pattern Verification ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for year in sorted(matrices.keys()):
        A = matrices[year]
        row_sums_by_power = []
        A_power = np.eye(A.shape[0])
        for k in range(8):
            if k > 0:
                A_power = A_power @ A
            row_sums_by_power.append(A_power.sum(axis=1).mean())
        ax.semilogy(range(8), row_sums_by_power, 'o-', linewidth=2, markersize=6,
                    label=str(year), color=colors.get(str(year), 'gray'))
    
    ax.set_xlabel('Power k', fontsize=12)
    ax.set_ylabel('Mean Row Sum of Aᵏ', fontsize=12)
    ax.set_title('Row Sum Decay Pattern\n'
                 '(Your observation: row sums of Aᵏ ≈ rᵏ where r = mean row sum of A)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'row_sum_pattern.png'), dpi=300)
    plt.close()
    print("  Saved row_sum_pattern.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BEA INPUT-OUTPUT DATA PROCESSOR")
    print("Real U.S. Economic Data — Summary Level (71 Industries)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    matrices = {}
    sector_names_dict = {}
    sector_codes_dict = {}
    all_errors = {}
    all_stats = []
    
    for year, filepath in sorted(FILES.items()):
        if not os.path.exists(filepath):
            print(f"\n⚠️  File not found: {filepath}")
            continue
        
        # Parse
        Z, x, codes, names = parse_bea_file(filepath, year)
        
        # Construct technology matrix
        A = construct_technology_matrix(Z, x)
        
        # Validate
        stats = validate(A, year, names)
        all_stats.append(stats)
        
        # Neumann series convergence
        print(f"\n  Neumann Series:")
        L, errors, k_converged = neumann_convergence(A)
        all_errors[year] = errors
        print(f"    Converged in {k_converged} terms")
        
        # Row sum pattern
        r = A.sum(axis=1).mean()
        L_row_mean = L.sum(axis=1).mean()
        predicted = 1.0 / (1.0 - r)
        print(f"\n  Row Sum Pattern:")
        print(f"    Mean row sum of A: {r:.6f}")
        print(f"    Predicted 1/(1−r): {predicted:.4f}")
        print(f"    Actual mean L row sum: {L_row_mean:.4f}")
        print(f"    Deviation: {abs(predicted - L_row_mean)/L_row_mean:.2%}")
        
        # Multipliers
        mult = compute_multipliers(A, L, names)
        
        print(f"\n  Top 5 by Backward Linkage:")
        for _, row in mult.head(5).iterrows():
            print(f"    {int(row['BL_rank']):>2}. {row['Sector'][:40]:<42} BL={row['BL_norm']:.3f}")
        
        print(f"\n  Top 5 by Forward Linkage:")
        fl_top = mult.sort_values('FL_norm', ascending=False).head(5)
        for _, row in fl_top.iterrows():
            print(f"    {int(row['FL_rank']):>2}. {row['Sector'][:40]:<42} FL={row['FL_norm']:.3f}")
        
        # Store
        matrices[year] = A
        sector_names_dict[year] = names
        sector_codes_dict[year] = codes
        
        # Save outputs
        np.save(os.path.join(OUTPUT_DIR, f'A_{year}.npy'), A)
        np.save(os.path.join(OUTPUT_DIR, f'Z_{year}.npy'), Z)
        np.save(os.path.join(OUTPUT_DIR, f'x_{year}.npy'), x)
        np.save(os.path.join(OUTPUT_DIR, f'L_{year}.npy'), L)
        
        pd.DataFrame(A, index=names, columns=names).to_csv(
            os.path.join(OUTPUT_DIR, f'A_{year}.csv'))
        mult.to_csv(os.path.join(OUTPUT_DIR, f'multipliers_{year}.csv'), index=False)
    
    # Save metadata
    with open(os.path.join(OUTPUT_DIR, 'sector_names.json'), 'w') as f:
        json.dump(sector_names_dict, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'sector_codes.json'), 'w') as f:
        json.dump(sector_codes_dict, f, indent=2)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY — U.S. TECHNOLOGY MATRICES")
    print(f"{'='*70}")
    print(f"{'Year':<6} {'n':<5} {'ρ(A)':<10} {'Col Sum':<22} {'Density':<10}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['year']:<6} {s['n']:<5} {s['rho']:<10.6f} "
              f"[{s['col_sum_min']:.3f}, {s['col_sum_max']:.3f}]{'':<6} {s['density']:<10.1%}")
    
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_stats.csv'), index=False)
    
    # Generate figures
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")
    generate_all_figures(matrices, sector_names_dict, all_errors)
    
    print(f"\n{'='*60}")
    print("WEEK 1 COMPLETE — REAL BEA DATA!")
    print(f"{'='*60}")
    print(f"All outputs: {OUTPUT_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"\nReady for Week 2: Network centrality analysis!")
    
    return matrices, sector_names_dict


if __name__ == '__main__':
    matrices, names = main()
