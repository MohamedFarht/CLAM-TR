# Anonymous Code Snapshot for Q1 NCA 2026 Paper

**Manuscript:** *Encoder × Aggregator Co-Selection for Cross-Cohort Prostate Cancer Grading: External Validation on TCGA-PRAD*
**Submission:** Neural Computing and Applications, Springer Nature (in review)
**Anonymous repo URL:** see manuscript Methods §2.6 reproducibility section

This snapshot contains the implementation, audit scripts, and figure-rendering scripts that reproduce every numerical result in the manuscript.

---

## Directory layout

```
anon_snapshot/
├── README.md                          # This file
├── final-code/
│   ├── 04-final-v20/                  # The CLAM-TR v20 RetNet-bestHP training notebooks (PANDA 5-fold CV)
│   │   ├── CLAM_TR_v20_FullScale_Colab.ipynb       # Full 9,128-slide training entry point (Colab)
│   │   ├── CLAM_TR_v20_FullScale_Local.ipynb       # Same, local-execution variant
│   │   ├── CLAM_TR_v20_SpatialAdd_bestHP.ipynb     # SpatialAdd aggregator variant
│   │   ├── CLAM_TR_v20_UNI_Colab.ipynb             # UNI v1 encoder variant
│   │   ├── CLAM_TR_v20_UNI_Colab_OPTIMIZED.ipynb   # Same with throughput optimizations
│   │   └── CLAM_TR_v20_UNI2h_Colab.ipynb           # UNI2-h encoder variant
│   └── 07-q1-sprint/                  # Q1 paper additions: TCGA-PRAD external validation
│       └── T2_2_phase3_tcga_prad_benign_extraction_colab.ipynb  # Benign-cohort GDC extraction
├── scripts/
│   ├── q1_fig_style.py                # Shared matplotlib style for figures 2-4
│   ├── q1_fig2_forest_vs_dspa.py      # Figure 2: forest plot vs DSPA-U-MIL baselines
│   ├── q1_fig3_per_site_forest.py     # Figure 3: per-site forest with bootstrap 95% CIs
│   ├── q1_fig4_benign_softmax.py      # Figure 4: R2 vs R5 benign ISUP histograms
│   ├── audit_dspa_real_benign.py      # Stage 3 external-validation audit
│   ├── audit_dspa_decision_rule_sweep.py  # Decision-rule sensitivity sweep (7 rules)
│   └── audit_per_site_subgroup.py     # Per-site QWK + over-prediction signature
└── data_pointers.md                   # Where to obtain PANDA + TCGA-PRAD raw data
```

---

## Reproducing the headline results

### Within-distribution (PANDA, 5-fold CV)

1. Download the PANDA dataset from Kaggle (https://www.kaggle.com/c/prostate-cancer-grade-assessment)
2. Run the noise-cleaning step described in `data_pointers.md` to obtain the 9,128-slide subset
3. Pre-extract UNI v1 patch features (https://huggingface.co/MahmoodLab/UNI; gated repo, accept terms)
4. Open `final-code/04-final-v20/CLAM_TR_v20_UNI_Colab_OPTIMIZED.ipynb` in Google Colab (A100 recommended for speed)
5. Run all cells; the notebook outputs per-fold QWK + per-grade accuracy + bootstrap 95% CI
6. Expected headline: κ_w = 0.9543 ± 0.0046 (RetNet-bestHP on UNI v1)

### External validation (TCGA-PRAD, zero-shot)

1. Download the TCGA-PRAD diagnostic + tissue slides from NIH GDC (https://portal.gdc.cancer.gov/projects/TCGA-PRAD); GDC manifests embedded in `final-code/07-q1-sprint/T2_2_phase3_tcga_prad_benign_extraction_colab.ipynb` Cell 4
2. Run the benign-extraction notebook (Colab, T4 GPU sufficient)
3. Run audit scripts in order:
   - `scripts/audit_dspa_real_benign.py` (Stage 3 external-validation primary audit)
   - `scripts/audit_dspa_decision_rule_sweep.py` (R1–R7 sensitivity sweep; produces Supplementary Table S2)
   - `scripts/audit_per_site_subgroup.py` (per-site QWK + over-prediction signature; produces Section 3.6)
4. Generate figures:
   - `python scripts/q1_fig2_forest_vs_dspa.py`
   - `python scripts/q1_fig3_per_site_forest.py`
   - `python scripts/q1_fig4_benign_softmax.py`
5. Expected headline: κ_w = 0.6381 (R2, full n=544 cohort)

### Decision rules

R1–R7 are defined in Supplementary §S8 and implemented in `audit_dspa_decision_rule_sweep.py`. R2 (slide-level expected-value at T=1, calibration-free) is the primary rule.

---

## Environment

- Python 3.10+
- PyTorch 2.0+
- torchvision, timm, transformers
- numpy, pandas, scikit-learn, scipy, matplotlib
- openslide-python (for WSI reading)
- huggingface-hub (for UNI v1 access)
- A100 (40 GB) GPU recommended for training; T4 sufficient for inference

A full `requirements.txt` is included in `scripts/`.

---

## Anonymity note

All author names, supervisor names, institutional affiliations, and acknowledgments have been removed from notebook headers and script comments for double-blind / single-blind review compatibility. The anonymous.4open.science upload service strips additional identifying metadata. Upon paper acceptance, the full deanonymised repository will be released on Zenodo with a citable DOI.
