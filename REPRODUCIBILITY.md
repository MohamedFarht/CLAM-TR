# Reproducibility — CLAM-TR

This file is the software-version manifest, random-seed policy, and compute-command
record that the study appendix says is "released together with the public code
repository." It complements `requirements.txt`, `best_lr.json`, and `data_pointers.md`.

## Random-seed policy
- `SEED = 42`, applied to `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and `torch.cuda.manual_seed(_all)`.
- `torch.backends.cudnn.deterministic = True`.
- `torch.backends.cudnn.benchmark` is **left at the PyTorch default (`False`)** — it is not explicitly set.
- **Determinism caveat:** the 5 folds are run concurrently via CUDA streams and use AMP mixed precision (`torch.amp.autocast('cuda')` + `GradScaler`); this can introduce minor run-to-run non-determinism in the low-order digits. Per-fold checkpoints are atomic (resume-from-crash safe).

## Compute environment
- **Main 9,128-slide factorial (both encoders):** Google Colab Pro+, **NVIDIA Tesla T4** (15.6 GB).
- **Exploratory LoRA branch only:** **NVIDIA A100 80 GB** (~179 GB system RAM). Not part of the primary factorial.
- **Software:** PyTorch 2.x, timm, HuggingFace Hub, OpenSlide, h5py, scikit-learn, scipy, statsmodels, matplotlib, seaborn. Full list with versions: `requirements.txt`. For a byte-exact lock, commit `pip freeze > requirements-lock.txt` from the Colab runtime.
- **Approx. runtime per single 5-fold experiment (T4):** UNI ~8–12 h; UNI2-h ~15–20 h.

## Exact compute command
Per study §2.6, "the exact training command is the recorded run itself, with cells
executed in order with seed = 42." The headline result is reproduced by:

1. Obtain the 9,128-slide noise-cleaned PANDA subset (see `data_pointers.md`).
2. Pre-extract UNI v1 features (gated `MahmoodLab/UNI`; accept terms on HuggingFace).
3. Open `final-code/04-final-v20/CLAM_TR_v20_UNI_Colab_OPTIMIZED.ipynb` in Colab (A100 recommended for speed; results were recorded on T4).
4. Run all cells in order. The notebook auto-loads the tuned learning rate `0.0002` from `best_lr.json` (cell 6 prints `Auto-loaded tuned LR: 0.0002`).
5. **Expected headline:** κ_w = 0.9543 ± 0.0051 (UNI `RetNet_bestHP`), pooled macro-F1 0.7748.

## Learning-rate selection
`best_lr.json` records the fold-0 LR search on `ABMIL_bestHP` over `[5e-5, 1e-4, 2e-4]`;
the selected `best_lr = 2e-4` for both UNI and UNI2-h is the value reported in the study.

## Determinism of the statistical analysis
Bootstrap CIs use B = 10,000 percentile resamples; BH-FDR via
`statsmodels.stats.multitest.fdrcorrection`; QWK via
`sklearn.metrics.cohen_kappa_score(weights='quadratic')`. The audit/figure scripts
under `scripts/` regenerate every reported number from the per-fold prediction dumps.
