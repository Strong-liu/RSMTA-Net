# Source provenance

This snapshot contains the author-supplied RSMTA-Net implementations for MI, MA, and WG. Model and training files are copied without changes. Absolute paths and subject-selection settings are retained.

| Release path | Local source directory |
| --- | --- |
| `RSMTA-MI/sta.py`, `RSMTA-MI/run_sta_net.py` | `RSMTA/RSMTA-MI/` |
| `RSMTA-MA/sta.py`, `RSMTA-MA/run_sta_net.py` | `RSMTA/RSMTA-MA/` |
| `RSMTA-WG/sta.py`, `RSMTA-WG/run_sta_net.py` | `RSMTA/RSMTA-WG/` |
| `RSMTA-MI/preprocessing/` | `RSMTA/STA-Net-main-MIdataset/preprocessing/` |
| `RSMTA-MA/preprocessing/` | `RSMTA/STA-Net-main-MAdataset/preprocessing/` |
| `RSMTA-WG/preprocessing/` | `RSMTA/STA-Net-main-WGdataset/preprocessing/` |
| `RSMTA-MI/run_time_window_accuracy.py` | `RSMTA/RSMTA-MI/` |

`SOURCE_MANIFEST.json` records the exact source path and both original and release SHA-256 checksums for each copied source or dependency file. The root README, dependency lists, ignore rules, and this notice were added for this release.

## Preprocessing syntax repair

The original MI `preprocessing.py` contained stray text after the closing print statement on line 428, followed by a duplicated obsolete fNIRS-processing block on lines 429–472. This prevented Python from parsing the file. In this release copy, the closing print statement is corrected and the obsolete duplicate block is removed. The resulting 428-line file matches the healthy MA preprocessing implementation apart from `TASK` and the selected participant. The original workspace file is retained unchanged. This repair does not modify the model, training, preprocessing parameters, paths, or selected participant.

Related alignment and preprocessing work includes:

Liu, M., Yang, B., Meng, L., Zhang, Y., Gao, S., Zan, P., and Xia, X. *STA-Net: Spatial–temporal alignment network for hybrid EEG-fNIRS decoding.* Information Fusion, article 103023 (2025). [Publisher record](https://www.sciencedirect.com/science/article/pii/S156625352500096X).

Dataset access and reuse remain subject to the original providers' terms. This notice records provenance and does not assign a new license to third-party material.
