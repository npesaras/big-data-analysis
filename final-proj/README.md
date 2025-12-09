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

### Installation

```bash
# Clone repository
git clone https://github.com/npesaras/big-data-analysis.git
cd big-data-analysis/final-proj

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Run Application

```bash
# Start Streamlit web app
uv run streamlit run main.py
```

Open http://localhost:8501 in your browser.

### Train Models

```bash
# Train classification models
uv run python scripts/train_classification.py
```

## Project Structure

```
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

## Model Performance

Best performing models (10-fold CV Recall):

| Model | Recall | Accuracy | Training Time |
|-------|--------|----------|---------------|
| Random Forest | 78.1% | 77.8% | ~2s |
| MLP | 75.2% | 76.1% | ~9s |
| Gaussian NB | 74.5% | 75.3% | <0.1s |
| Logistic Regression | 74.1% | 75.8% | <0.2s |

*Recall prioritized to minimize false negatives in medical diagnosis*

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

### Programmatic Usage

```python
from src.config import Config
from src.data_cleaning import load_diabetes_data
from src.models import create_classification_pipeline
from src.utils import load_model

# Load configuration
print(f"Data directory: {Config.DATA_DIR}")

# Load and preprocess data
X, y = load_diabetes_data(str(Config.DIABETES_DATA))

# Create and train model
pipeline = create_classification_pipeline()
pipeline.fit(X, y)

# Make predictions
predictions = pipeline.predict(X)
```
## Key Technologies

- **Python 3.13+**: Core language
- **scikit-learn 1.7.2**: Machine learning algorithms
- **Streamlit 1.51.0**: Web application framework
- **Pandas 2.3.3**: Data manipulation
- **Plotly 6.5.0**: Interactive visualizations
- **uv**: Fast Python package manager (Rust-based)

## Usage Examples

### Load and Preprocess Data

```python
from src.data_cleaning import load_diabetes_data
from src.pre_processing import preprocess_pipeline

# Load raw data
df = load_diabetes_data()

# Complete preprocessing (zeros → NaN → impute → scale)
df_processed, preprocessors = preprocess_pipeline(df)
```

### Train Multiple Models

```python
from src.model_selection import create_all_models
from src.model_training import train_all_models

# Create 9 models
models = create_all_models()

# Train with 10-fold CV
results = train_all_models(X_train, y_train, cv_folds=10, scoring='recall')

# View results
for name, metrics in results.items():
    print(f"{name}: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
```

### Make Predictions

```python
from src.utils import load_model, load_preprocessors

# Load trained model and preprocessors
model = load_model('models/best_model.joblib')
preprocessors = load_preprocessors('models/preprocessors.joblib')

# New patient data
patient = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 25,
    'Insulin': 100,
    'BMI': 28.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 33
}

# Preprocess and predict
X_new = pd.DataFrame([patient])
X_processed = apply_preprocessing(X_new, preprocessors)
prediction = model.predict(X_processed)
probability = model.predict_proba(X_processed)

print(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")
print(f"Confidence: {probability[0][1]:.1%}")
```

---
metrics = evaluate_classification_model(pipeline, X, y)
```

### Utilities (`src.utils`)

```python
from src.utils import load_model, save_model, setup_logging

# Model persistence
model = load_model(filepath)
save_model(model, filepath)

# Logging setup
setup_logging(level='INFO', log_file='training.log')
```

---

## Development

### Environment Setup

```bash
# Create virtual environment
uv venv

# Activate environment
## Contributing

We welcome contributions! Guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes with clear messages
4. Push and open a Pull Request

Follow PEP 8, add type hints, update docs, maintain modular architecture.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Authors

**Nilmar Pesaras** - *ITD105 Big Data Analytics Project*

## Acknowledgments

- PIMA Indians Diabetes Dataset from UCI ML Repository
- scikit-learn team for ML algorithms
- Streamlit for web framework
- Astral team for uv package manager
