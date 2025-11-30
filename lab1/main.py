import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
        color: #262730;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the student performance dataset"""
    try:
        # Try to load the dataset from the provided URL or local file
        # You can replace this with the actual path to your student-mat.csv file
        df = pd.read_csv('data/student-mat.csv', sep=';')
        return df
    except FileNotFoundError:
        st.error("Please download the student-mat.csv file and place it in the same directory as this script.")
        st.info("Download link: https://drive.google.com/drive/folders/1Bz9q37BB20PJSWsdGH__cshZGfPKSpHd?usp=sharing")
        return None

def display_dataset_info(df):
    """Display basic dataset information"""
    st.subheader("📋 Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Data types
    st.subheader("📊 Data Types")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(dtype_df, use_container_width=True)

def display_summary_statistics(df):
    """Display summary statistics for numeric columns"""
    st.subheader("📈 Summary Statistics")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_stats = df[numeric_cols].describe()
    st.dataframe(summary_stats, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    st.subheader("🔥 Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    st.pyplot(fig)

    return correlation_matrix

def create_boxplot(df):
    """Create boxplot for numeric features"""
    st.subheader("📦 Boxplot Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['G1', 'G2', 'G3']]  # Exclude target variables

    if len(numeric_cols) > 0:
        # Select columns for boxplot
        selected_cols = st.multiselect(
            "Select features for boxplot:",
            numeric_cols,
            default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
        )

        if selected_cols:
            fig, ax = plt.subplots(figsize=(12, 8))
            df[selected_cols].boxplot(ax=ax)
            plt.title('Boxplot of Selected Features')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

def create_interactive_scatter(df):
    """Create interactive scatter plot using Plotly"""
    st.subheader("🎯 Interactive Scatter Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Select X-axis:", numeric_cols, index=0)
    with col2:
        y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=1)

    # Color by gender if available
    color_col = 'sex' if 'sex' in df.columns else None

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color_col,
        title=f"{x_axis} vs {y_axis}",
        hover_data=df.columns.tolist()
    )

    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

def analyze_correlations(correlation_matrix):
    """Analyze correlations with exam scores"""
    st.subheader("🎓 Correlation Analysis with Exam Scores")

    exam_cols = ['G1', 'G2', 'G3']
    available_exam_cols = [col for col in exam_cols if col in correlation_matrix.columns]

    if available_exam_cols:
        for exam_col in available_exam_cols:
            st.write(f"**{exam_col} Correlations:**")
            exam_corr = correlation_matrix[exam_col].drop(exam_col).sort_values(key=abs, ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Top 5 Positive Correlations:")
                top_positive = exam_corr[exam_corr > 0].head()
                for feature, corr in top_positive.items():
                    st.write(f"• {feature}: {corr:.3f}")

            with col2:
                st.write("Top 5 Negative Correlations:")
                top_negative = exam_corr[exam_corr < 0].head()
                for feature, corr in top_negative.items():
                    st.write(f"• {feature}: {corr:.3f}")

            st.write("---")

def create_additional_visualizations(df):
    """Create additional visualizations for grade booster"""
    st.subheader("📊 Additional Visualizations")

    # Bar chart for categorical features
    if 'sex' in df.columns:
        st.write("**Gender Distribution:**")
        gender_counts = df['sex'].value_counts()
        fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                    title="Gender Distribution",
                    labels={'x': 'Gender', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

    # Pair plot for selected numeric features
    st.write("**Pair Plot Analysis:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exam_cols = ['G1', 'G2', 'G3']
    other_numeric = [col for col in numeric_cols if col not in exam_cols]

    selected_features = st.multiselect(
        "Select features for pair plot:",
        other_numeric,
        default=other_numeric[:4] if len(other_numeric) >= 4 else other_numeric
    )

    if len(selected_features) >= 2:
        pair_plot_cols = selected_features + exam_cols
        pair_plot_cols = [col for col in pair_plot_cols if col in df.columns]

        if len(pair_plot_cols) >= 2:
            fig = px.scatter_matrix(
                df[pair_plot_cols],
                title="Pair Plot of Selected Features"
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">📊 Student Performance Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df is None:
        st.stop()

    # Sidebar for filtering
    st.sidebar.header("🔍 Data Filters")

    # Gender filter
    if 'sex' in df.columns:
        gender_filter = st.sidebar.selectbox("Filter by Gender:", ["All"] + list(df['sex'].unique()))
        if gender_filter != "All":
            df = df[df['sex'] == gender_filter]

    # Age filter
    if 'age' in df.columns:
        age_range = st.sidebar.slider("Age Range:",
                                     int(df['age'].min()),
                                     int(df['age'].max()),
                                     (int(df['age'].min()), int(df['age'].max())))
        df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]

    # Parental education filter
    if 'Medu' in df.columns and 'Fedu' in df.columns:
        medu_filter = st.sidebar.selectbox("Mother's Education:", ["All"] + sorted(df['Medu'].unique().tolist()))
        fedu_filter = st.sidebar.selectbox("Father's Education:", ["All"] + sorted(df['Fedu'].unique().tolist()))

        if medu_filter != "All":
            df = df[df['Medu'] == medu_filter]
        if fedu_filter != "All":
            df = df[df['Fedu'] == fedu_filter]

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dataset Info", "📈 Statistics", "🔥 Correlations", "📊 Visualizations", "🎯 Interactive Analysis"])

    with tab1:
        display_dataset_info(df)
        st.subheader("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        display_summary_statistics(df)

    with tab3:
        correlation_matrix = create_correlation_heatmap(df)
        analyze_correlations(correlation_matrix)

    with tab4:
        create_boxplot(df)
        create_additional_visualizations(df)

    with tab5:
        create_interactive_scatter(df)

        # Answer the lab questions
        st.subheader("❓ Lab Questions Analysis")

        st.write("**a. Which features have the highest correlation with the final exam scores (G1, G2, G3)?**")
        if 'G1' in df.columns and 'G2' in df.columns and 'G3' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exam_cols = ['G1', 'G2', 'G3']
            other_cols = [col for col in numeric_cols if col not in exam_cols]

            if other_cols:
                correlations = {}
                for exam in exam_cols:
                    for feature in other_cols:
                        corr = df[exam].corr(df[feature])
                        correlations[f"{feature} vs {exam}"] = abs(corr)

                top_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
                for feature_pair, corr in top_correlations:
                    st.write(f"• {feature_pair}: {corr:.3f}")

        st.write("**b. How does study time correlate with exam performance?**")
        if 'studytime' in df.columns and 'G3' in df.columns:
            study_corr = df['studytime'].corr(df['G3'])
            st.write(f"Study time vs Final Grade (G3) correlation: {study_corr:.3f}")

            # Create a visualization
            fig = px.scatter(df, x='studytime', y='G3',
                           title="Study Time vs Final Grade",
                           labels={'studytime': 'Study Time (hours)', 'G3': 'Final Grade'})
            st.plotly_chart(fig, use_container_width=True)

        st.write("**c. What insights can you draw from the boxplot?**")
        st.write("The boxplot shows the distribution, outliers, and quartiles of numeric features, helping identify:")
        st.write("• Data spread and central tendency")
        st.write("• Presence of outliers")
        st.write("• Skewness in the data")
        st.write("• Feature ranges and variability")

        st.write("**d. How does gender impact the final exam score?**")
        if 'sex' in df.columns and 'G3' in df.columns:
            gender_stats = df.groupby('sex')['G3'].agg(['mean', 'median', 'std', 'count']).round(2)
            st.dataframe(gender_stats)

            # Create gender comparison chart
            fig = px.box(df, x='sex', y='G3', title="Final Grade Distribution by Gender")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
