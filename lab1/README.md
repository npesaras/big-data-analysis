# Student Performance Analysis Dashboard

A comprehensive Exploratory Data Analysis (EDA) dashboard for student exam performance built with Streamlit, as part of the ITD105 Big Data Analytics course.

## ✨ Features

- 📊 **Interactive Dashboard**: Complete EDA with multiple visualization types
- 🔍 **Data Exploration**: Dataset overview, statistics, and correlations
- 📈 **Visual Analytics**: Heatmaps, boxplots, scatter plots, and pair plots
- 🎯 **Lab Analysis**: Automated analysis of course lab questions
- 🔧 **Smart Filtering**: Filter by gender, age, and parental education
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- **Python 3.12 or higher**
- **uv** package manager (modern Python packaging)

### 1. Install uv

**On macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

**Verify installation:**

```bash
uv --version
```

### 2. Clone and Setup Project

```bash
# Navigate to the lab1 directory
cd lab1

# Install dependencies using uv
uv pip install -r requirements.txt
```

### 3. Download Dataset

Download the `student-mat.csv` dataset from:
[Google Drive Link](https://drive.google.com/drive/folders/1Bz9q37BB20PJSWsdGH__cshZGfPKSpHd?usp=sharing)

Place the file in the `data/` directory.

### 4. Run the Application

```bash
# Run with uv
uv run streamlit run main.py
```

The app will open at `http://localhost:8501` in your browser.

## 📋 Dashboard Overview

### Navigation Tabs

1. **📋 Dataset Info**
   - Dataset summary (rows, columns, memory usage)
   - Data types and missing values overview
   - Preview of first 10 rows

2. **📈 Statistics**
   - Comprehensive statistical summary
   - Mean, median, standard deviation, quartiles

3. **🔥 Correlations**
   - Interactive correlation heatmap
   - Feature correlation analysis with exam scores
   - Top positive/negative correlations

4. **📊 Visualizations**
   - Customizable boxplots for numeric features
   - Gender distribution charts
   - Pair plot analysis

5. **🎯 Interactive Analysis**
   - Interactive scatter plots with Plotly
   - Automated lab questions analysis
   - Gender impact on performance

### Interactive Filters

- **Gender**: Filter by male/female students
- **Age Range**: Select age range with slider
- **Parental Education**: Filter by mother/father education levels

## 🔍 Lab Questions Analysis

The dashboard automatically addresses the course lab requirements:

### A. Feature Correlations with Exam Scores

Identifies which features have the strongest relationships with G1, G2, and G3 exam scores.

### B. Study Time Impact

Analyzes how study time correlates with academic performance.

### C. Boxplot Insights

Explains distribution patterns, outliers, and data spread from boxplot visualizations.

### D. Gender Differences

Compares exam performance statistics between male and female students.

## 🛠️ Technical Details

### Built With

- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **Matplotlib/Seaborn** - Static plots
- **NumPy** - Numerical computing

### Key Features

- ⚡ **Performance Optimized**: Uses Streamlit caching for fast loading
- 🎨 **Professional UI**: Custom CSS styling and responsive layout
- 🛡️ **Error Handling**: Graceful handling of missing data
- 📊 **Data Validation**: Automatic data type checking and conversion

## 📁 Project Structure

```text
lab1/
├── main.py                 # Main Streamlit application
├── pyproject.toml          # Project configuration
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── data/
│   └── student-mat.csv    # Dataset (download separately)
├── docs/
│   └── laboratory-analysis.md
└── assets/                # Static assets
```

## 🔧 Troubleshooting

### Common Issues

**"Dataset not found" error:**

- Ensure `student-mat.csv` is placed in the `data/` directory
- Verify the filename matches exactly

**Import errors:**

```bash
# Reinstall dependencies
uv pip install -r requirements.txt
```

**Port already in use:**

- Streamlit will automatically use the next available port
- Check terminal output for the correct URL

**Python version issues:**

- Ensure you're using Python 3.12 or higher
- Check with: `python --version`

### Performance Tips

- The app uses caching to optimize performance
- For large datasets, consider filtering data in the sidebar first
- Close unused browser tabs to free memory

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Plotly Python](https://plotly.com/python/)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure the dataset file is correctly placed
4. Review error messages in the terminal

## 📝 Course Information

**Course**: ITD105 - Big Data Analytics
**Lab Exercise**: #1 - Student Performance Analysis
**Objective**: Build an interactive EDA dashboard using Streamlit
