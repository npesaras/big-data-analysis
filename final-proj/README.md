# Diabetes Classification System

Machine learning pipeline for diabetes prediction using the PIMA Indians Diabetes Dataset. Implements 9 classification algorithms with comprehensive preprocessing, cross-validation, and a Streamlit web interface.

## Features

- **9 ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN, SVM, AdaBoost, Perceptron, MLP
- **Robust Preprocessing**: Zero handling, median imputation, StandardScaler normalization
- **Stratified 10-Fold CV**: Reliable performance estimation with confidence intervals
- **Streamlit Web App**: Interactive patient data input with real-time predictions
- **Modular Architecture**: Clean separation of concerns across 10 Python modules
- **Production Ready**: Model persistence, preprocessor saving, reproducible pipeline

## Quick Start

### Deployed URL

open this url on your browser

```url
https://diabetes-classification-model.streamlit.app/
```

### Installation

```bash
# Clone repository
git clone https://github.com/npesaras/big-data-analysis.git
cd big-data-analysis/final-proj

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# change directory to final-proj
cd big-data-analysis/final-proj

# Install dependencies

uv sync

# use uv environment
source .venv/bin/activate
```

### Run Application

```bash
# Start Streamlit web app
streamlit run main.py
```

## Project Structure

```txt
final-proj/
├── main.py                  # Streamlit web application
├── data/
│   └── diabetes.csv         # PIMA Indians Diabetes dataset (768 samples)
├── src/                     # Core modules (10 files)
│   ├── config.py            # Configuration and hyperparameters
│   ├── data_cleaning.py     # Data loading and inspection
│   ├── exploratory_data_analysis.py  # Visualizations
│   ├── pre_processing.py    # Preprocessing pipeline
│   ├── data_splitting.py    # Train/test split, CV folds
│   ├── model_selection.py   # 9 algorithm definitions
│   ├── model_training.py    # Training procedures
│   ├── model_evaluation.py  # Metrics and comparison
│   └── utils.py             # Helper functions
├── models/                  # Trained model artifacts (.joblib)
├── docs/                    # Technical documentation
│   ├── 00_installation.md   # Setup guide
│   ├── 01_data_collection.md
│   ├── 02_exploratory_analysis.md
│   ├── 03_preprocessing.md
│   ├── 04_data_splitting.md
│   ├── 05_model_selection.md
│   ├── 06_model_training.md
│   ├── 07_model_evaluation.md
│   └── 08_api_reference.md
└── scripts/                 # Training scripts
    └── train_classification.py
```

---

## Dataset

**PIMA Indians Diabetes Database**

- **Samples**: 768 female patients of Pima Indian heritage
- **Features**: 8 clinical measurements (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Target**: Binary classification (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution**: 500 non-diabetic (65%) vs 268 diabetic (35%)
- **Challenge**: Missing values encoded as zeros (Insulin: 49%, Skin Thickness: 30%)


## Technical Documentation

Comprehensive guides available in `docs/`:

- **[Installation Guide](docs/00_installation.md)** - uv setup, environment configuration
- **[Data Collection](docs/01_data_collection.md)** - Dataset loading, quality inspection
- **[Exploratory Analysis](docs/02_exploratory_analysis.md)** - Visualizations, correlations
- **[Preprocessing](docs/03_preprocessing.md)** - Zero handling, imputation, scaling
- **[Data Splitting](docs/04_data_splitting.md)** - Stratified splits, cross-validation
- **[Model Selection](docs/05_model_selection.md)** - Algorithm comparison (9 models)
- **[Model Training](docs/06_model_training.md)** - Training procedures, CV
- **[Model Evaluation](docs/07_model_evaluation.md)** - Metrics, confusion matrices
- **[API Reference](docs/08_api_reference.md)** - Complete function documentation

---

## Usage

### Web Application

The Streamlit application provides diabetes risk prediction:

#### Diabetes Prediction

- Input patient diagnostic measurements
- Get real-time diabetes risk assessment
- View confidence scores and clinical interpretation

## Contributing

We welcome contributions! Guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes with clear messages
4. Push and open a Pull Request

Follow PEP 8, add type hints, update docs, maintain modular architecture.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- PIMA Indians Diabetes Dataset from UCI ML Repository
- scikit-learn team for ML algorithms
- Streamlit for web framework
- Astral team for uv package manager
