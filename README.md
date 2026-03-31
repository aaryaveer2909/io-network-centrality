# Spectral Network Analysis of U.S. Input-Output Tables (2007–2024)

**Identifying Systemically Critical Economic Sectors: A Spectral Network Analysis of U.S. Input-Output Tables**

## Overview

This repository contains the complete computational pipeline for analysing the U.S. economy's inter-industry structure using network centrality measures, spectral decomposition, and shock propagation simulation on Bureau of Economic Analysis (BEA) input-output data.

The project compares six measures of sector importance — backward linkage, forward linkage, eigenvector centrality, PageRank, betweenness centrality, and Katz centrality — and validates each against a hypothetical extraction benchmark. It also includes case studies of the Global Financial Crisis (2007→2012) and COVID-19 pandemic (2017→2022).

## Key Findings

- **Forward linkage** is the strongest predictor of systemic sector importance (Spearman ρ ≈ 0.93)
- **Betweenness centrality** identifies supply-chain bottleneck sectors invisible to traditional multiplier analysis
- The U.S. economy's **spectral radius declined** from 0.491 (2012) to 0.457 (2024), indicating shorter effective supply chains
- A **rank-15 SVD approximation** captures 90% of the technology matrix's structure across all years
- **Katz centrality with α = 1 is mathematically equivalent to the Leontief forward linkage**, unifying network science and input-output economics

## Data

Download BEA Use Tables (Before Redefinitions, Producers' Prices, Summary level) from:
https://apps.bea.gov/iTable/?reqid=150&step=1

Select: Input-Output → Make-Use-Imports (Before Redefinitions) → Use of Commodities by Industries (Producers' Prices) → Summary level

Download for years: 2007, 2012, 2017, 2022, 2024

## Pipeline

Run scripts in order:

### Step 1: Data Parsing (Week 1-2)
```bash
python bea_parser.py
```
Parses BEA Excel files → constructs 71×71 technology matrices → computes Leontief inverses → validates Neumann series convergence → generates backward/forward linkage rankings.

**Input:** BEA Excel files in `/mnt/user-data/uploads/`  
**Output:** `week1/outputs/` — A matrices (.npy, .csv), Leontief inverses, multiplier rankings, convergence plots, eigenvalue spectra

### Step 2: Network Centrality & Shock Simulation (Weeks 3-5)
```bash
python network_analysis.py
```
Constructs weighted directed graphs → computes 4 centrality measures → Spearman correlation analysis → hypothetical extraction for all 66 private sectors → cascade failure simulation → GFC and COVID case studies.

**Input:** Technology matrices from Step 1  
**Output:** `week3_5/outputs/` — centrality rankings, correlation heatmaps, extraction impacts, cascade results, case study data

### Step 3: Spectral Analysis (Week 6)
```bash
python spectral_analysis.py
```
Full eigenvalue decomposition → SVD low-rank approximation → NMF economic archetypes → Perron vector analysis → random matrix theory comparison → formal row-sum pattern verification.

**Input:** Technology matrices from Step 1  
**Output:** `week6/outputs/` — spectral statistics, SVD errors, NMF factors, Perron vectors, all figures

## Requirements

```
numpy
scipy
pandas
networkx
matplotlib
seaborn
openpyxl
scikit-learn
```

Install: `pip install numpy scipy pandas networkx matplotlib seaborn openpyxl scikit-learn`

## File Structure

```
├── bea_parser.py           # Step 1: BEA data → technology matrices
├── network_analysis.py     # Step 2: Centrality + shock simulation
├── spectral_analysis.py    # Step 3: Spectral decomposition
├── generate_paper.js       # Paper generation (docx-js)
├── README.md
├── week1/outputs/          # Technology matrices, multipliers, convergence data
├── week3_5/outputs/        # Centrality rankings, extraction results, case studies
└── week6/outputs/          # Spectral statistics, SVD, NMF, Perron vectors
```

## Citation

If you use this code or methodology, please cite:

```
Aaryaveer. (2026). Identifying Systemically Critical Economic Sectors: 
A Spectral Network Analysis of U.S. Input-Output Tables (2007–2024). 
[Preprint/Journal Name].
```

## Author

**Aaryaveer**  
Bachelor of Data Science, SP Jain School of Global Management, Sydney, Australia  
GitHub: github.com/aaryaveer2909

## License

MIT License
