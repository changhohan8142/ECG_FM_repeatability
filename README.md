# Within-recording consistency improves decision repeatability in adapted electrocardiogram foundation models

Code for the study of the same title: construction of the datasets, adaptation of three ECG foundation models (ECGFounder, HuBERT-ECG, ECG-FM),
the three remedies (distillation from a teacher trained on short windows, mixed reading, penalized classification head), the evaluation, and every table and figure of the paper.

> Authors, citation and licence: to be added on publication.

## What the study does

A fine-tuned ECG model reads a 10-s recording. If the recording is cut into its two adjacent 5-s halves and the model reads each half as an ECG, the patient, the electrodes and the device are the same,
so the two decisions should agree. They often do not. The code measures how often the decision changes, between the two halves of one recording and between two recordings of the same patient,
and it implements three remedies that seek agreement of the output across overlapping windows of the segment being read:

| Step | Remedy | What it does |
|---|---|---|
| Training | Distillation | a student that reads the whole segment is trained to match the mean output of a teacher over seven 2.5-s windows |
| Reading | Mixed reading | probability = 0.1 × one read of the segment + 0.9 × mean over 25 overlapping 2.5-s windows |
| Head | Penalized head | a logistic regression refitted with a penalty on directions of the embedding that change between two shifted views of the same segment |

## Contents

Everything is in `notebooks/`, to be run in this order. There are no Python modules: settings and shared functions live in notebook 00, which every other notebook loads with `%run`.

| Notebook | Content | Needs |
|---|---|---|
| `00_setup_and_settings.ipynb` | environment, **global settings** (the master folder and all paths), constants of the study, shared functions (signal processing, the three foundation models, training, penalized head, reading) | – |
| `01_data_download_and_preprocessing.ipynb` | where to obtain PTB-XL, MIMIC-IV-ECG, CODE-15% and EchoNext and under which conditions; download; filtering; cached arrays | internet, disk space |
| `02_training_and_evaluation_sets.ipynb` | exclusions, waveform check, unlabelled training pool, evaluation sets, pairs of recordings | 01 |
| `03_model_adaptation.ipynb` | pretrained weights; fine-tuning of the baseline, the teacher and the student of each foundation model, and of the comparison models | GPU, 02 |
| `04_reading_and_penalized_head.ipynb` | single read, window readings, penalized head, Monte Carlo dropout; outputs of every fine-tuned model for every evaluation recording | GPU, 03 |
| `05_analysis.ipynb` | decision thresholds, decision change, AUROC, direction agreement, paired bootstrap; summary tables | 04 |
| `06_computational_cost.ipynb` | time of adaptation and of reading, baseline against the final setting | GPU |
| `07_tables.ipynb` | Tables 1–2 and Supplementary Tables 1–4 | 02, 05 |
| `08_figures.ipynb` | Figs. 1–5 and Supplementary Figs. 1–3 | 02, 04, 05, 06 |

In notebooks 07 and 08 every table and figure has its own heading (`## Table 2`, `## Supplementary Fig. 1`, …) with the caption of the paper above the code.

## Getting started

1. **Environment** (Python 3.10; the versions used for the paper are pinned):

   ```bash
   conda env create -f environment.yml
   conda activate ecg-fm-repeatability
   python -m ipykernel install --user --name ecg-fm-repeatability
   jupyter lab
   ```

   or `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` followed by `pip install -r requirements.txt`.

2. **Master folder.** Choose a folder with enough free space for the data, the models and the results, and either set the environment variable `ECGREP_MASTER` to it
   or edit the line `MASTER = ...` in the cell *Global settings* of `notebooks/00_setup_and_settings.ipynb`. Nothing is read or written outside this folder.

3. **Data and pretrained models.** They are not part of this repository. Notebook 01 explains for each dataset where it comes from, what access it requires and where the files go;
   notebook 03 does the same for the pretrained weights of the three foundation models.

4. **Run the notebooks in order.** Long steps skip what already exists, so they can be interrupted and resumed. Notebooks 03, 04 and 06 have a switch `QUICK_TEST` for a short functional check.

## Data

| Dataset | Used for | Source | Access |
|---|---|---|---|
| PTB-XL 1.0.3 | training labels, validation, test | PhysioNet | open |
| MIMIC-IV-ECG 1.0 | unlabelled training pool, test, pairs of recordings | PhysioNet | open |
| MIMIC-IV 3.1 (patients table) | age and sex of the MIMIC-IV-ECG patients in Table 1 (optional) | PhysioNet | credentialed users, data use agreement |
| CODE-15% | test, pairs with a change of label | Zenodo | open |
| EchoNext 1.1.1 | test (echocardiographic wall thickness) | PhysioNet | registered users, data use agreement |

The exact files, versions, licences and conditions are given in notebook 01; read the terms of each dataset on its own page before you use it.
Notebook 01 downloads what is openly accessible; EchoNext and the MIMIC-IV patients table have to be obtained by hand.
All four datasets are needed to build the table of evaluation recordings in notebook 02 (the MIMIC-IV patients table is optional).
Disk space: about 80 GB of raw data and 60 GB of cached arrays when only the MIMIC-IV-ECG recordings that are examined are downloaded.

## Pretrained models

| Model | Source | Licence stated by its authors |
|---|---|---|
| ECGFounder | github.com/PKUDigitalHealth/ECGFounder (code), Hugging Face `PKUDigitalHealth/ECGFounder` (weights) | MIT |
| HuBERT-ECG (base) | Hugging Face `Edoardo-Coppola/hubert-ecg-base` | CC BY-NC 4.0 |
| ECG-FM | Hugging Face `wanglab/ecg-fm` | MIT |

No code or weights of these projects are copied into this repository. Fine-tuned models are written to `MASTER/models/finetuned` by notebook 03 and are read from there by the later notebooks.

## Reproducibility

- Sets are drawn with fixed seeds, so that notebook 02 gives the same training pool and evaluation sets every time.
- Training uses fixed seeds, but results on a graphics card are not bit-identical between runs or computers. Small differences in the last digits are to be expected.
- Confidence intervals come from a bootstrap with a fixed seed (`BOOT_SEED` in notebook 00). The intervals in the paper were calculated with other resamples of the same kind,
  so limits can differ in the last digit from those printed in the paper; point estimates do not depend on the resamples.
- Hardware used: one NVIDIA RTX 3080 (10 GB). Fine-tuning one model takes 3–46 min; the 58 fine-tuned models of the paper take about 14 h of GPU time.
  Notebooks 03, 04 and 06 need a graphics card; the others run on a processor.
- The time to read a single segment (notebook 06) is limited by the processor rather than by the graphics card, and it depends on which processor cores the process runs on.
- The notebooks are stored without outputs.

## Repository layout

```
├── README.md
├── environment.yml        conda environment with the versions used
├── requirements.txt       the same versions for pip
└── notebooks/             00 … 08, see above
```
