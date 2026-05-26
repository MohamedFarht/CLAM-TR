# Data Pointers

## PANDA (training cohort)

- Source: Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge
- URL: https://www.kaggle.com/c/prostate-cancer-grade-assessment
- License: Creative Commons Attribution-ShareAlike 4.0
- Original size: 10,616 slides
- Noise-cleaning step: filter out the slides flagged as noisy in the dataset providers' supplementary list (Bulten et al., *Nature Medicine* 2022). Resulting cohort: 9,128 slides spanning ISUP grades 0–5.

## TCGA-PRAD (external test cohort)

- Source: NIH Genomic Data Commons
- URL: https://portal.gdc.cancer.gov/projects/TCGA-PRAD
- License: NIH open-access (no controlled-access components used)
- Cancer-positive cohort: 426 Diagnostic Slides (sample type 01, Primary Tumor; DX-prefixed)
- Benign cohort: 118 Tissue Slides (sample type 11, Solid Tissue Normal; TS-prefixed)
- Exact GDC REST API queries are embedded in the manuscript Supplementary §S7

## UNI v1 encoder

- Source: Mahmood Lab (Chen et al., *Nature Medicine* 2024)
- URL: https://huggingface.co/MahmoodLab/UNI
- License: gated repository; researchers must accept terms before download
- Used as frozen feature extractor (1024-dim ViT-L/16 embeddings)
