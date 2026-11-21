# Flood Impact Scoring from Post-Hurricane Aerial Imagery

**Group:** LB: Learning Beyond  
**Leader:** Lyle Mora  
**Members:** Bhargav Garre Venkata
**Course:** DS 4002, Section 002 — Professor Loreto Alonzi  

We evaluate semantic segmentation models to identify flooded regions and compute flood impact severity scores from post-hurricane aerial imagery. The study uses the FloodNet high-resolution aerial imagery dataset [1] (totaling approximately 13 GB across training, validation, and test splits). Our primary goal is to achieve strong mean Intersection over Union (mIoU) and flooded-class segmentation performance while accounting for spatial and near-duplicate data leakage across evaluation splits.

## Repository contents

`Data/` holds dataset documentation, provenance metadata, class dictionaries, near-duplicate pair logs, and access instructions for the external 13 GB dataset. `Scripts/` holds source code for data preprocessing, near-duplicate verification, model training, evaluation metrics, and the interactive Gradio demo. `Output/` holds generated figures, metric tables, and preliminary model predictions. Supporting documentation is in [Data/README.MD](Data/README.MD), [Scripts/README.MD](Scripts/README.MD), and [Output/README.MD](Output/README.MD).

### 1. Software and platform

The data processing, model training, and evaluation pipeline was run on Google Colab / Linux (Ubuntu), Python 3.10+, with PyTorch, `segmentation_models_pytorch` [2], NumPy, pandas, SciPy, scikit-learn, Matplotlib, OpenCV, and Gradio [3]. Install the core analysis and modeling dependencies using:

```bash
python -m pip install torch torchvision segmentation-models-pytorch numpy pandas scipy scikit-learn matplotlib opencv-python gradio
```

For the notebook and interactive exploration, also install `jupyterlab` and `ipykernel`. These provide IPython and a notebook interface. Use python3 instead of python if required by your environment. GPU acceleration (CUDA via Google Colab T4/A100) is recommended for model training and batch inference; evaluation scripts and metric computations run smoothly on CPU or GPU. No additional review or test files are required to execute the project.

### 2. Documentation map

```text
Project_3/
├── Data/
│   ├── README.MD
│   ├── data_dictionary.csv
│   ├── near_duplicate_pairs.csv
│   ├── provenance.json
│   ├── quality_report.json
│   └── raw/
│       ├── image_metadata.json
│       └── raw_data_pointers.txt
├── LICENSE
├── Output/
│   ├── Dataset_Tables/
│   │   ├── overall_evaluation_metrics.csv
│   │   └── per_class_iou_summary.csv
│   ├── EDA_Figures/
│   │   └── segmentation_examples.png
│   ├── README.MD
│   ├── Reports/
│   │   ├── provenance.json
│   │   └── quality_report.json
│   └── predictions/
├── README.MD
├── REFERENCES.MD
└── Scripts/
    ├── README.MD
    ├── 01_data_preprocessing.py
    ├── 02_near_duplicate_check.py
    ├── 03_train_segmentation.py
    ├── 04_evaluate_predictions.py
    └── 05_gradio_demo.py
```

The preprocessing and training scripts write generated files directly into `Output/`. Its existing subfolders hold organized copies; rerunning does not automatically replace those copies. Raw dataset inputs remain accessed externally via Dropbox unless explicitly copied as described below.

### 3. Reproduce the results

1. **Open a terminal in the repository root**, containing this README and the three project folders. Install the dependencies listed above. Reruns replace generated files with the same names.

2. **Download and link the dataset.** Download the 13 GB dataset archive from the [project Dropbox repository](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1) and extract it into `Data/raw/` (or mount it via Google Drive in Colab). See [Data/README.MD](Data/README.MD) for expected directory structure and access verification.

3. **Run preprocessing and near-duplicate filtering.** Run:

   ```bash
   python Scripts/01_data_preprocessing.py
   python Scripts/02_near_duplicate_check.py
   ```

   This verifies image-mask alignment, builds train/validation/test splits, and flags the 72 candidate near-duplicate image pairs to construct the leakage-aware test subset.

4. **Train and evaluate the segmentation models.** Run:

   ```bash
   python Scripts/03_train_segmentation.py
   python Scripts/04_evaluate_predictions.py
   ```

   This trains the segmentation architecture (using `segmentation_models_pytorch`) [2] on the training split, validates after each epoch, and saves model checkpoints and test predictions to `Output/`. Evaluation scripts compute mean IoU (mIoU), per-class IoU, predicted vs. ground-truth percent flooded area, and flood impact category agreement across both the full test set and the leakage-aware test subset. Results are exported to `Output/Dataset_Tables/`.

5. **Launch the Gradio demo and run EDA.** Launch python -m jupyterlab, open the EDA notebook in Scripts/, select your active python environment, restart the kernel, and run all cells. To run the web demo, execute:

    ```bash
   python Scripts/05_gradio_demo.py
   ```

This launches a local/Colab web interface [3] allowing users to upload post-hurricane aerial imagery and view predicted segmentation masks and flood impact severity scores.

## Expected modeling results

| Model / Split | Mean IoU (mIoU) | Flooded Class IoU | Percent Flooded MAE | Category Agreement |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Segmentation (Full Test) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Baseline Segmentation (Leakage-Aware Subset) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

*Note: Full model training and final test-set evaluation metrics are actively in progress for MI4. Preliminary validation outputs and scoring scripts are fully operational in `Scripts/04_evaluate_predictions.py`.*

## Decisions and limitations

* **External Data Hosting:** Due to GitHub file size limits, the 13 GB raw dataset cannot be committed directly to the repository. Reproducibility relies on the frozen Dropbox archive and path configurations documented in [Data/README.MD](Data/README.MD).
* **Near-Duplicate Spatial Leakage:** Post-hurricane UAV imagery frequently contains overlapping flight paths and near-identical frames. Following manual review of candidate near-duplicate pairs, evaluation metrics are reported on both the standard test split and a leakage-aware subset with near-duplicates removed.
* **MI3 Scope:** In accordance with MI3 milestones, repository structure, metadata documentation, and evaluation pipelines are fully established, while model hyperparameter tuning, final test-set evaluations, and the Gradio web demo remain in active development toward MI4.

## License and attribution

The existing [LICENSE](LICENSE) provides MIT terms for project software. Third-party imagery and annotations retain their own source conditions and attribution requirements [1], as documented in [Data/README.MD](Data/README.MD) and [REFERENCES.MD](REFERENCES.MD). GitHub access and repository visibility are managed by the group.

## References

Numbers apply to this README; Data source_ref tags follow the Data README's numbering. The extended bibliography is in [REFERENCES.MD](REFERENCES.MD).

[1] Rahnemoonfar, M., et al., “FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding,” *IEEE Access*, vol. 9, pp. 89644–89654, 2021.

[2] Yakubovskiy, P., “Segmentation Models PyTorch,” GitHub Repository, 2019. [Online]. Available: https://github.com/qubvel-org/segmentation_models.pytorch

[3] Abid, A., et al., “Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild,” arXiv:1906.02569, 2019.
