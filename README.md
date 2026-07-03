# Credit Score Classification — UAS Project

Prediksi kategori *Credit Score* nasabah (`Good` / `Standard` / `Poor`) dari data historis
keuangan & perilaku kredit bulanan (`data/data_A.csv`, dataset "Overhead" Credit Score Classification).

```
project/
├── data/
│   └── data_A.csv                  # dataset mentah
├── notebooks/
│   └── eksperiment.ipynb           # BAGIAN A - EDA, cleaning, OOP pipeline, 5 model comparison
├── src/                            # BAGIAN B - training pipeline modular (OOP) + MLflow
│   ├── ingest.py                   # load raw data + structural cleaning
│   ├── preprocessing.py            # split + skew-aware imputation/scaling/encoding
│   ├── training.py                 # abstract BaseModelTrainer + subclasses per model
│   ├── evaluation.py               # metrics, confusion matrix, model comparison
│   └── pipeline.py                 # orkestrasi end-to-end + MLflow tracking + .pkl artifacts
├── artifacts/                      # dihasilkan oleh src/pipeline.py
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── target_encoder.pkl
│   ├── best_model_name.txt
│   ├── model_comparison.csv
│   └── classification_report_*.txt
├── tests/
│   └── test_inference.py           # BAGIAN C - test case per kelas (Good/Standard/Poor)
├── app_streamlit.py                # BAGIAN C - web app deployment (Streamlit)
├── requirements.txt
└── mlflow.db                       # MLflow tracking store (SQLite, dibuat otomatis)
```

## Bagian A — Eksperimen (notebook)

Buka dan jalankan `notebooks/eksperiment.ipynb`. Notebook ini membangun pipeline OOP
(`FeatureSelector` → `EDA` → `DataCleaner` → `Splitter` → `Preprocessor` → `TreeModelZoo` →
`ModelEvaluator`), membandingkan **5 model tree-based** (Decision Tree, Random Forest, Extra Trees,
Gradient Boosting, XGBoost), dan menyimpulkan **3 model terbaik** yang dipakai di Bagian B.

## Bagian B — Training Pipeline Lokal + MLflow

```bash
pip install -r requirements.txt
cd src
python pipeline.py
```

Ini akan:
1. Ingest & bersihkan `data/data_A.csv` (`ingest.py`)
2. Split + preprocessing skew-aware (`preprocessing.py`)
3. Melatih 3 model terbaik dari Bagian A: **RandomForest, XGBoost, GradientBoosting** (`training.py`,
   pakai abstract base class `BaseModelTrainer` + inheritance per model)
4. Evaluasi & bandingkan (`evaluation.py`), log semua parameter/metric/model ke **MLflow**
5. Pilih model terbaik & simpan `.pkl` ke folder `artifacts/`

Untuk melihat hasil tracking MLflow di browser:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

lalu buka `http://localhost:5000`.

## Bagian C — Deployment (Streamlit)

```bash
streamlit run app_streamlit.py
```

App akan memuat `artifacts/best_model.pkl`, `preprocessor.pkl`, dan `target_encoder.pkl`, lalu
menampilkan form input data nasabah, hasil prediksi kelas, dan **bar chart probability distribution**
untuk ketiga kelas.

### Testing

```bash
python tests/test_inference.py
```

Menjalankan 3 test case nyata (satu representatif per kelas `Good`/`Standard`/`Poor`) melalui
`preprocessor.pkl` + `best_model.pkl`, memverifikasi probabilitas valid (jumlah = 1) dan prediksi
sesuai label asli.

### Deploy ke Streamlit Community Cloud via GitHub

1. Push seluruh folder project ini (termasuk folder `artifacts/` yang sudah berisi `.pkl`) ke repo GitHub.
2. Login ke [share.streamlit.io](https://share.streamlit.io), pilih repo tersebut.
3. Set **Main file path** ke `app_streamlit.py`.
4. Streamlit Cloud akan otomatis membaca `requirements.txt` untuk instalasi dependency.
