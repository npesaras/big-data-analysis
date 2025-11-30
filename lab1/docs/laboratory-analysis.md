**Nilmar T. Pesaras**

Exploratory Data Analysis (EDA) of Student Exam Performance

**Questions:**

a.  **Which features have the highest correlation with the final exam
    scores (G1, G2, G3)?**

The top 5 features correlated with each exam score are:

- **G1**: failures (0.355), Medu (0.205), Fedu (0.190),
  studytime (0.161), goout (0.149)

- **G2**: failures (0.356), Medu (0.216), Fedu (0.165), goout (0.162),
  traveltime (0.153)

- **G3**: failures (0.360), Medu (0.217), age (0.162), Fedu (0.152),
  goout (0.133)

**Key insight**: Past class failures show the strongest
correlation (negative) with all exam scores. Parental education
(especially mother\'s education) also shows consistent positive
correlation with performance.

b.  **How does study time correlate with exam performance?**

- Study time has a **weak positive correlation (0.098)** with
  final grade (G3).

**Key insight**: While studying more does slightly improve performance,
the weak correlation suggests that study quality, learning efficiency,
or other factors may be more important than just time spent studying.

c.  **What insights can you draw from the boxplot?**

The boxplot reveals important distribution patterns:

- **Absences**: Highly skewed with 15 outliers (max 75), indicating some
  students miss significantly more school than others

- **Goout**: Evenly distributed (no outliers), showing balanced social
  habits across students

- **Exam scores (G1, G2, G3)**: More concentrated distributions with few
  outliers, though G2 has 13 outliers including very low scores

- **Overall**: Most features show normal distributions with outliers
  highlighting at-risk students who may need intervention

d.  **How does gender impact the final exam score?**

Gender-based performance statistics for G3:

- **Female (F)**: mean 9.97, median 10.0, std 4.62 (n=208)

<!-- -->

- **Male (M)**: mean 10.91, median 11.0, std 4.50 (n=187)

**Key insight**: Male students score approximately **0.94** points
higher on average (on a 0-20 scale). However, both groups show similar
variability (std \~4.5-4.6), suggesting the difference is modest and
both genders have comparable performance ranges.
