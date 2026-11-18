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

**CHANGE EVERYTHING BELOW THIS IS FROM PROJECT 2**

## Expected modeling results

| Model | Test MAE | Test RMSE |
| --- | ---: | ---: |
| SARIMAX without weather | 1.086120 | 1.395554 |
| SARIMAX with earlier weather | 1.090465 | 1.400223 |
| Constant median (predict 1) | 1.114911 | 1.592991 |
| Constant mean | 1.118596 | 1.429067 |
| Seven-day seasonal naive | 1.476060 | 1.969327 |

The selected order is **(1,0,1)** with seasonal order **(1,0,1,7)** and an intercept. The final test constants are **mean = 1.538732394** and **median = 1**. Validation constants are mean = 1.523505249 and median = 1, calculated before 2023.

Both SARIMAX models beat both constant benchmarks and seasonal naive on test MAE. However, the weather model does not beat the no-weather model, so the original goal is **not met**. The weather-minus-no-weather MAE difference is approximately **+0.004344402** crashes/day, with a reported 95% block-bootstrap interval of **[−0.000654253, 0.009597203]**. This does not demonstrate improvement from the lagged weather features.

## Decisions and limitations

The source records count reported city crashes, not all emergencies or risk per trip. Reporting/publication timing is imperfectly known, so this is retrospective forecasting. SARIMAX is a Gaussian model applied to small counts; negative test point forecasts are clipped at zero for both models, and positive forecasts are not rounded. Validation MAE currently uses raw forecasts. Stationarity/invertibility are not enforced; convergence alone does not guarantee stability. Residual Ljung–Box results are descriptive and are not adjusted for fitted AR/MA degrees of freedom.

The implemented sensitivity check trains on 2017–2019 and forecasts 2020–2021. This is a documented deviation from MI2's planned exclusion of 2020–2021 when fitting through 2023; the two analyses are not equivalent. Full-period EDA is not evidence that the test years were unseen during exploration. Predictive associations do not establish causation.

## License and attribution

The existing [LICENSE](LICENSE) provides MIT terms for project software. Third-party data retain their own source conditions: preserve VDOT terms [1] and Open-Meteo/ERA5 attribution and applicable CC BY 4.0 requirements [2], [4], [5]. Data/README.MD documents unresolved VDOT redistribution conditions. GitHub access and repository visibility are managed by the group.

## References

Numbers apply to this README; Data source_ref tags follow the Data README's numbering. The extended bibliography is in [REFERENCES.MD](REFERENCES.MD).

[1] Virginia Department of Transportation, “Full_Crash,” Virginia Roads, ArcGIS REST Services Directory. Accessed: Oct. 21, 2026. [Online]. Available: https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services/Full_Crash/FeatureServer

[2] Open-Meteo, “Historical Weather API.” Accessed: Oct. 21, 2026. [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api

[3] statsmodels developers, “statsmodels.tsa.statespace.sarimax.SARIMAX,” statsmodels documentation. Accessed: Oct. 21, 2026. [Online]. Available: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

[4] Open-Meteo, “Terms & Privacy.” Accessed: Oct. 21, 2026. [Online]. Available: https://open-meteo.com/en/terms

[5] Creative Commons, “Attribution 4.0 International (CC BY 4.0).” Accessed: Oct. 21, 2026. [Online]. Available: https://creativecommons.org/licenses/by/4.0/
