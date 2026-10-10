# Campaign Response Prediction — Data Science & AI/ML

A Jupyter Notebook project that demonstrates data preparation, statistical analysis, classification, clustering, and artificial neural network (ANN) modeling for a binary campaign-response target.

## Project Structure

Keep this README and `Exam.ipynb` in the project root. The notebook reads the input dataset from `data/raw/set_d.csv` and saves generated artifacts under `outputs/`.

```text
Project 3/
├── data/
│   └── raw/
│       └── set_d.csv
├── outputs/
│   ├── figures/
│   │   ├── engagement_histogram.png
│   │   ├── confusion_matrix.png
│   │   ├── kmeans_silhouette.png
│   │   └── ann_loss_curve.png
│   ├── models/
│   ├── ann_test_predictions.csv
│   ├── ann_training_info.csv
│   ├── cluster_profiles.csv
│   ├── clustered_fit.csv
│   ├── covariance_matrix.csv
│   ├── eigenvalues.csv
│   ├── final_model_comparison.csv
│   ├── inference_results.csv
│   ├── kmeans_diagnostics.csv
│   ├── model_metrics.csv
│   ├── preprocessing_audit.csv
│   ├── splits.csv
│   ├── statistics_summary.csv
│   └── test_predictions.csv
├── Exam.ipynb
└── README.md
```

*The exact output files depend on successfully running the corresponding notebook cells.*

## Project Overview

The notebook demonstrates the following workflow:

1. Load the dataset from `data/raw/set_d.csv`.
2. Inspect and clean the data, including handling duplicate records.
3. Create fit, validation, and test partitions.
4. Summarize engagement and perform statistical inference comparing groups.
5. Calculate covariance and eigenvalue summaries.
6. Impute missing numeric values, encode categorical features, and scale predictors.
7. Train and evaluate a baseline classifier and Logistic Regression.
8. Run K-Means clustering and compare cluster counts using silhouette scores.
9. Train an ANN for binary response prediction and track training/validation loss.
10. Export model metrics, predictions, diagnostics, figures, and model artifacts to `outputs/`.

## Screenshots

The notebook saves all figures in `outputs/figures/`. GitHub renders these images directly from the repository when the paths below are preserved.

### 1. Engagement Distribution

![Engagement Distribution](outputs/figures/engagement_histogram.png)

### 2. Logistic Regression — Test Confusion Matrix

![Logistic Regression Test Confusion Matrix](outputs/figures/confusion_matrix.png)

### 3. K-Means Silhouette Scores

![K-Means Silhouette Scores](outputs/figures/kmeans_silhouette.png)

### 4. ANN Training and Validation Loss

![ANN Training and Validation Loss](outputs/figures/ann_loss_curve.png)

## Technologies

- Python
- Pandas and NumPy
- SciPy
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Models and Analysis

- **Baseline classification:** most-frequent baseline for comparison.
- **Logistic Regression:** supervised binary classification.
- **Artificial Neural Network:** dense neural network with ReLU hidden layers and a sigmoid output for binary classification.
- **K-Means:** unsupervised clustering with silhouette-score diagnostics.
- **Statistical analysis:** descriptive summaries, Welch's t-test, covariance, and eigenvalue analysis.
- **Preprocessing:** missing-value imputation, categorical encoding, and feature scaling.

## Dataset

The notebook expects a CSV file at:

```text
data/raw/set_d.csv
```

The notebook can create a sample CSV at this location if the file is missing. If you have your own dataset, place it at this exact path before running the notebook; the updated notebook preserves an existing CSV instead of replacing it.

The dataset should contain the columns expected by the notebook, including:

| Column | Role |
|---|---|
| `record_id` | Record identifier; not intended as a predictor |
| `visits` | Numeric feature |
| `recency` | Numeric feature |
| `engagement` | Numeric feature |
| `spend` | Numeric feature |
| `group` | Categorical feature |
| `response` | Binary target |

Use data that is appropriately collected and authorized for the intended analysis.

## Installation and Usage

### 1. Open the project directory

Make sure `Exam.ipynb`, `data/`, and `outputs/` are in the same project root.

### 2. Install dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn tensorflow jupyter
```

TensorFlow compatibility varies by Python version and operating system. If installation fails, use a supported environment and follow TensorFlow's official installation guidance.

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `Exam.ipynb` and run the cells in order. Run the notebook with the project root as the working directory so relative paths resolve correctly.

## Output Files

The notebook writes results to `outputs/`, including:

| Output | Description |
|---|---|
| `outputs/splits.csv` | Record partition assignments |
| `outputs/statistics_summary.csv` | Descriptive engagement statistics |
| `outputs/inference_results.csv` | Group-comparison test summary |
| `outputs/covariance_matrix.csv` | Covariance matrix |
| `outputs/eigenvalues.csv` | Eigenvalue and variance-share summary |
| `outputs/preprocessing_audit.csv` | Preprocessing checks |
| `outputs/model_metrics.csv` | Baseline and Logistic Regression metrics |
| `outputs/test_predictions.csv` | Classification test predictions |
| `outputs/kmeans_diagnostics.csv` | K-Means diagnostics |
| `outputs/cluster_profiles.csv` | Cluster profile summaries |
| `outputs/final_model_comparison.csv` | Final model comparison |
| `outputs/ann_test_predictions.csv` | ANN test predictions |
| `outputs/ann_training_info.csv` | ANN training summary |
| `outputs/figures/` | Saved PNG figures |
| `outputs/models/` | Saved preprocessing and ANN model artifacts |

## Evaluation Metrics

The notebook reports or uses common classification metrics:

- **Accuracy:** proportion of predictions that are correct.
- **Precision:** proportion of predicted positive cases that are true positives.
- **Recall:** proportion of actual positive cases that are correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts of correct and incorrect predictions by class.
- **Silhouette score:** a measure of how separated the K-Means clusters are.

Review the exported CSV files for the actual numeric results from your run. The screenshots illustrate model behavior but should not replace test-set metrics.

## Notes

- Keep the file names and folder paths unchanged unless you update the notebook accordingly.
- All PNG figures should be saved under `outputs/figures/`.
- Saved models and preprocessing objects should be kept under `outputs/models/`.
- Do not commit private, sensitive, or unauthorized customer data to a public repository.
- Add your name and course details if required.

## Author

**Name:** Add your name here  
**Project:** Campaign Response Prediction — Data Science & AI/ML
