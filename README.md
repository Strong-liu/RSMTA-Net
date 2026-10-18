# RSMTA-Net

TensorFlow implementation of **RSMTA-Net: A Receptive-Field Spatial Modeling and State-Space Temporal Alignment Network for Hybrid EEG-fNIRS Decoding**.

RSMTA-Net combines spatial receptive-field modeling with state-space enhancement of fNIRS candidate features before EEG-guided temporal attention. This repository contains separate implementations for motor imagery (MI), mental arithmetic (MA), and word generation (WG).

## Repository structure

```text
RSMTA-MI/
  sta.py                         Model definition; factory: sta_net()
  run_sta_net.py                 MI training and evaluation
  run_time_window_accuracy.py    Optional MI time-window experiment
  preprocessing/                Six preprocessing stages
RSMTA-MA/
  sta.py
  run_sta_net.py
  preprocessing/
RSMTA-WG/
  sta.py
  run_sta_net.py
  preprocessing/
requirements.txt                Training dependency versions
requirements-preprocessing.txt  Additional preprocessing dependencies
SOURCE_MANIFEST.json            Source mapping and SHA-256 checksums
NOTICE.md                       Related work and source provenance
```

The task-specific Python sources retain their original absolute data paths and current subject-selection settings. Configure these settings in your own checkout before running. The MI preprocessing copy has one documented syntax repair to remove an accidentally duplicated obsolete tail; see [NOTICE.md](NOTICE.md). Data, trained weights, and generated experiment outputs are not included.

## Environment

The supplied projects target Python 3.9 and TensorFlow/Keras 2.10. Use an existing compatible TensorFlow GPU environment or create a separate environment:

```bash
python -m pip install -r requirements.txt
```

For preprocessing from raw recordings, also install:

```bash
python -m pip install -r requirements-preprocessing.txt
```

The additional preprocessing file is a dependency list rather than a complete environment lock. The original dependency lists are retained under each task's `preprocessing/requirements.original.txt`. Preprocessing includes interactive MNE/Matplotlib inspection and therefore needs a graphical display. A fresh installation and full dataset training have not been re-executed as part of preparing this source snapshot.

## Data

Obtain the recordings from their original providers and follow their access conditions:

| Task | Public dataset | Binary labels in the supplied pipeline |
| --- | --- | --- |
| MI | [Open access dataset for simultaneous EEG and NIRS BCI](https://doc.ml.tu-berlin.de/hBCI/) | Left-hand versus right-hand imagery |
| MA | [Open access dataset for simultaneous EEG and NIRS BCI](https://doc.ml.tu-berlin.de/hBCI/) | Mental arithmetic versus baseline |
| WG | [Simultaneous acquisition of EEG and NIRS during cognitive tasks](https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/) | Word generation versus baseline |

The MI/MA dataset contains 29 participants; the WG dataset contains 26. MI and MA use different sessions of the same public dataset. The scripts retain the task-specific processing of the two recording formats.

### Model input

Each participant is stored in one `.npz` file under a task's `model_input` directory.

| Key | Shape | Meaning |
| --- | --- | --- |
| `eeg` | `(N, 16, 16, 600, 1)` | EEG spatial grids and time samples |
| `fnirs` | `(N, 11, 16, 16, 30, 2)` | Eleven fNIRS candidates; last axis is HbO/HbR |
| `label` | `(N, 2)` | One-hot binary labels |
| `sample_session` | `(N,)` | Required by MI/MA; session IDs 0, 1, 2 |

MI/MA preprocessing also retains trial-level session metadata. The WG training script uses the original ordering of 600 windows: three consecutive sessions of 200 windows each. Each trial contributes ten EEG windows. The training scripts multiply the stored fNIRS input by `1e3`; do not apply this scaling a second time to an already scaled input.

## Prepare inputs from raw data

Skip this section if compatible `model_input` files are already available. Run the following stages in the task's `preprocessing` directory, in order:

```bash
python load_mat.py
python preprocessing.py
python epoch.py
python to3d.py
python window.py
python model_input.py
```

The intermediate directories are `mat2array`, `preprocessed`, `epoch`, `d3`, `window`, and `model_input`.

Before starting:

1. **MI/MA:** set `RAW_ROOT` consistently in all preprocessing files and in `run_sta_net.py`. Keep `TASK = "MI"` or `TASK = "MA"` consistent with the directory. Raw EEG/NIRS folders and montage files remain necessary for the relevant stages. Review each stage's `sub_ids` selection: `preprocessing.py` currently selects participant 05 for MI and participant 16 for MA.
2. **WG:** configure the absolute input/output directories in each stage. `preprocessing.py` currently selects `subject_no = 'VP001'`; process the other participants using the same procedure. Create the output directories used by later WG stages before running them, since some original stages assume they already exist.
3. Complete the interactive EEG artifact/ICA inspection when prompted. MI/MA convert raw fNIRS intensity to optical density and haemoglobin concentrations; WG begins with HbO/HbR recordings and uses its own filtering procedure.

## Train and evaluate

Run from the chosen task directory so that `from sta import sta_net` resolves to that task's model:

```bash
cd RSMTA-MI
python run_sta_net.py
```

Use the same commands from `RSMTA-MA` or `RSMTA-WG` for the other tasks. Each script starts training when executed; it is not an import-only library.

### Required local configuration

| Task | Configuration to review before execution |
| --- | --- |
| MI | Set `RAW_ROOT`. Near the bottom, enable the existing `for sub_id in MAIN_SUBJECT_IDS:` block to run participants 01–04 and 06–29. The supplied snapshot has this block commented out and enables only participant 05's two valid sessions. |
| MA | Set `RAW_ROOT`. The script processes the `.npz` files in that task's `model_input` directory. |
| WG | Set `subject_path`, the `Logger(...)` output path near the top, and the result paths in the script. Ensure parent directories exist. Keep the `model_input` directory limited to participant input files. |

For MI's complete cohort, retain the participant 05 call as well as enabling the main participant loop. Participant 05 has no valid samples in session 0 after preprocessing, so only sessions 1 and 2 are evaluated for that participant. The intended complete session-wise evaluation therefore has 86 MI folds, 87 MA folds, and 78 WG folds when all expected participants are present.

### Training settings in the supplied scripts

The outer evaluation holds out one session of the same participant. MI/MA use explicit session metadata; WG uses its original contiguous session ordering. The scripts preserve their different validation procedures: MI and WG sample validation windows from the development sessions; MA selects complete trials within those sessions.

Training uses two stages, with epoch limits of 300 and 200. The second stage continues the same model and optimizer on all development-session data, including the validation subset, while retaining validation monitoring and early stopping. Consult each task's callbacks for its exact patience and scheduling settings. MI/MA set Python, NumPy, and TensorFlow seeds to 42; WG fixes the validation sampling seed but does not set an equivalent TensorFlow model seed.

Evaluation CSVs and confusion-matrix figures are saved to the paths defined in each script. The final fused prediction is `class_output`; `eeg_output` is the auxiliary EEG prediction. The model factory keeps its historical function name `sta_net()`.

The optional MI script `run_time_window_accuracy.py` launches a separate training experiment and writes its own time-window results. It removes its previous time-window CSV at startup; use it only when rerunning that experiment intentionally.

## Related work

Please acknowledge the original dataset publications and relevant methodological work when using this implementation. The alignment methodology is related to *STA-Net: Spatial–temporal alignment network for hybrid EEG-fNIRS decoding*, Information Fusion (2025), article 103023: [publisher page](https://www.sciencedirect.com/science/article/pii/S156625352500096X). See [NOTICE.md](NOTICE.md) for the source mapping.

The RSMTA-Net manuscript title is provided above; a final publication DOI is not assigned in this snapshot.
