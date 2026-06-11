# ML Experiment Orchestrator

A lightweight, fully-configurable Python framework for running end-to-end machine learning
experiments — from raw data ingestion through model evaluation — driven entirely by a
single YAML configuration file.

---

## Project Structure

```
ml_orchestrator/
├── src/
│   ├── data/        # DataLoader, DataCleaner, DataSplitter
│   ├── models/      # ModelTrainer, Evaluator
│   └── utils/       # YAML config reader, dual-output logger
├── configs/
│   └── experiment.yaml   ← edit this to configure your run
├── tests/           # pytest unit tests (2+ per module)
├── notebooks/       # Exploration notebooks
├── run.py           # Main entry point
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py --config configs/experiment.yaml
```

> **No real data?** No problem. If the data file specified in the config does not exist,
> `run.py` automatically generates a 300-row synthetic CSV dataset so you can
> verify the pipeline end-to-end immediately.

---

## Configuring `experiment.yaml`

```yaml
data:
  filepath: "data/sample.csv"   # Path to your dataset (CSV, JSON, or Parquet)
  target_col: "target"          # Column name to predict

cleaning:
  fill_nulls: "mean"            # mean | median | mode | drop
  drop_duplicates: true
  fix_dtypes: true              # Coerce object columns to numeric where possible

splitting:
  test_size: 0.2                # Fraction of full dataset for test set
  val_size: 0.1                 # Fraction of full dataset for validation set
  random_seed: 42               # For reproducibility
  stratify: true                # Set false for regression tasks

model:
  type: "logistic_regression"   # logistic_regression | random_forest | decision_tree
                                # random_forest_regressor | ridge | lasso | svc | svr
  params: {}                    # Extra kwargs forwarded to the sklearn constructor
  save_path: "outputs/model.joblib"

evaluation:
  report_path: "outputs/report.json"
```

---

## Running Tests

```bash
pytest tests/
```

The test suite exercises `DataLoader`, `DataCleaner`, and `DataSplitter` with small
in-memory DataFrames — no network access or real files required.

---

## Example Output

```
================================================
  ML EXPERIMENT ORCHESTRATOR — RESULTS SUMMARY
================================================
  task                       classification
  accuracy                   0.916667
  f1_weighted                0.915611
  precision_weighted         0.915278
  recall_weighted            0.916667
  confusion_matrix           [[28, 2], [3, 27]]
================================================
```

A full run log is written to `run.log` and a JSON metrics report is saved to
`outputs/report.json`.

---

## Adding a New Model

1. Import your sklearn estimator in `src/models/trainer.py`.
2. Add a new key → class entry to `_MODEL_REGISTRY`.
3. Set `model.type` to your new key in `experiment.yaml`.

No other changes needed.
