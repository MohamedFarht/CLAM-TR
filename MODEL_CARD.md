# Model Card — CLAM-TR (RetNet-bestHP)

A model card following the structure of Mitchell et al. (2019). All numbers
are reproduced from the recorded experimental records; see `CITATION.cff`,
`REPRODUCIBILITY.md`, and `best_lr.json` for provenance.

## Model details
- **Name:** CLAM-TR — Retention-Enhanced Dual-Branch Aggregation. Headline configuration: `RetNet_bestHP` on the UNI v1 encoder.
- **Type:** Slide-level multiple-instance-learning (MIL) aggregator for whole slide images (WSI). The frozen foundation-model encoder is *not* trained.
- **Architecture:** A CLAM gated-attention branch and a retention branch operate over the TopK = 128 patches selected by gated attention; the two branch slide embeddings are combined by **additive fusion** (`M_clam + M_ret`) and classified by a single `Linear(512, 6)` head (six ordinal ISUP classes G0–G5). `num_heads = 4`, `hidden_dim = 512`.
- **Encoders (frozen):** UNI v1 (ViT-L/16, 1024-d) and UNI2-h (ViT-H/14, 1536-d), from the gated `MahmoodLab/UNI` and `MahmoodLab/UNI2-h` releases.
- **Best hyperparameters (bestHP):** `lr = 2e-4` (tuned via the fold-0 LR search in `best_lr.json`; the pre-tuning default dict value was `1e-4`), `weight_decay = 1e-5`, `dropout = 0.15`, `attention_temperature = 0.7`, retention γ-bank `[0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995]`, up to 50 epochs with early stopping, `seed = 42`.
- **Version:** v20 (thesis-final). **License:** see `CITATION.cff` (TODO — confirm; CLAM upstream is GPL-3.0).

## Intended use
- **Primary:** research on slide-level ISUP/Gleason grade-group prediction from prostate WSIs, and on encoder × aggregator co-selection in computational pathology.
- **Out of scope / NOT intended:** clinical diagnosis or treatment decisions. This is **not a medical device** and has not been validated for clinical deployment. Outputs must not be used for patient care.

## Training data
- **Dataset:** noise-cleaned **PANDA** prostate-biopsy cohort, **9,128 slides** (derived from the 10,616-slide raw release: Karolinska 5,758 + Radboud 4,858), labelled by ISUP grade group G0–G5.
- **Split:** 5-fold stratified cross-validation on the Kaggle 1st-place split; **slide-level isolation** by construction (patient-level grouping is not verifiable from the released metadata — see thesis §2.1.3).
- **Features:** pre-extracted and cached (UNI `.pt`, fp16, 1024-d; UNI2-h `.h5`, 1536-d).

## Evaluation
- **Within-distribution (PANDA, 5-fold CV):** QWK **0.9543 ± 0.0051** (UNI `RetNet_bestHP`); accuracy 0.8253 ± 0.0148; pooled **macro-F1 0.7748**. UNI2-h best: `SpatialMultRetNet_bestHP` QWK 0.9530 ± 0.0033. (± is the across-fold sample SD, ddof = 1.)
- **Provider-stratified (PANDA):** Radboud QWK 0.9532 / macro-F1 0.793; Karolinska QWK 0.9412 / macro-F1 0.717 (weaker per-class balance on Karolinska).
- **Cross-cohort, zero-shot (TCGA-PRAD):** QWK **0.638** with calibration-free expected-value decoding (Q1 follow-up; cross-cohort distribution shift).
- **Metric:** quadratic-weighted Cohen κ (`sklearn.metrics.cohen_kappa_score(weights='quadratic')`).

## Limitations
- **Small fold count (n = 5):** several pairwise contrasts show large Cohen's *d* with non-significant FDR-corrected *q* — an under-powered regime, read as a sample-size statement, not proof of architectural superiority.
- **Per-site heterogeneity:** zero-shot per-site κ_w ranges from −0.07 to +0.92 across contributing sites; aggregate cross-cohort numbers hide site-level variance.
- **Calibration collapse on benign:** under cancer-tuned temperature scaling, benign cases are misclassified; calibration-free decoding is required for the benign cohort.
- **Single training distribution:** trained only on PANDA; cross-cohort generalisation is partial.
- **Patient-level grouping** in the CV split cannot be verified from released metadata (slide-level isolation only).

## Ethical considerations
- **Research-use-only**; not for clinical use. Trained on the publicly released PANDA dataset; no patient data were collected by the authors.
- **Generative-AI assistance** during the accompanying write-up is disclosed in the thesis (§ AI-use methods note + KTU FBE-FR-O22 form); no scientific decision, number, or figure was produced by an LLM.
