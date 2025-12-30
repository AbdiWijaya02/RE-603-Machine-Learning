# Exploratory Data Analysis - COVID-19 Dataset

Comprehensive exploratory data analysis demonstrating best practices for data exploration and visualization.

## 📋 Project Information

- **Notebook:** `EDA - COVID-19 Dataset Analysis.ipynb`
- **Topic:** Exploratory Data Analysis (EDA)
- **Dataset:** COVID-19 Synthetic Dataset
- **Focus:** Data exploration, statistical analysis, visualization
- **Complexity:** Intermediate

## 🎯 Learning Objectives

In this project, you will learn:
- Teknik comprehensive EDA
- Data exploration best practices
- Advanced data visualization
- Statistical analysis
- Data quality assessment
- Pattern discovery
- Report generation

## 📊 Dataset

**Synthetic COVID-19 Dataset** (synthetic_covid19_data.xlsx) - Dataset sintetis untuk tujuan pembelajaran

- **Format:** Excel file (.xlsx)
- **Type:** Tabular data / Time series data
- **Likely Features:** 
  - Date/Time information
  - COVID-19 cases, deaths, recoveries
  - Demographic information
  - Geographic location
  - Testing data
  - Vaccination data (if applicable)

### Data Characteristics:
- Synthetic (generated untuk learning)
- Multiple columns/features
- Various data types (numeric, categorical, date)
- Time series component

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Data loading & manipulation |
| **numpy** | Numerical operations |
| **matplotlib** | Visualization |
| **seaborn** | Statistical visualization |
| **scikit-learn** | Statistical functions |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook EDA\ -\ COVID-19\ Dataset\ Analysis.ipynb
```

### Google Colab
1. Upload notebook & dataset ke Colab
2. Install dependencies jika perlu
3. Run cells dari atas ke bawah

## 📝 Inti Notebook - EDA Framework

### 1. **Load Libraries & Data**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_excel('synthetic_covid19_data.xlsx')
```

### 2. **First Look at Data**
```python
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())
```

### 3. **Data Quality Assessment**

#### Missing Values
```python
# Check missing values
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Percentage': missing_percent
})

print(missing_df)

# Visualize
plt.figure(figsize=(10, 6))
missing_data.plot(kind='barh')
plt.title('Missing Values by Column')
plt.show()
```

#### Duplicate Rows
```python
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Remove if necessary
df = df.drop_duplicates()
```

#### Data Type Validation
```python
# Check data types
print(df.dtypes)

# Identify numeric, categorical, date columns
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns
```

### 4. **Statistical Summary**

#### Univariate Analysis
```python
# Summary statistics untuk numeric columns
print(df[numeric_cols].describe().T)

# Distribution analysis
for col in numeric_cols:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"{col}: Skewness={skewness:.2f}, Kurtosis={kurtosis:.2f}")
```

#### Categorical Analysis
```python
# Value counts untuk categorical columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
```

### 5. **Data Visualization**

#### Distribution Plots
```python
# Histograms untuk numeric features
fig, axes = plt.subplots(len(numeric_cols)//2 + 1, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col], bins=30, edgecolor='black')
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

#### Box Plots untuk Outliers
```python
fig, axes = plt.subplots(len(numeric_cols)//2 + 1, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'Box Plot of {col}')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.show()
```

#### Count Plots untuk Categorical
```python
fig, axes = plt.subplots(len(categorical_cols)//2 + 1, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[idx])
    axes[idx].set_title(f'Count of {col}')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### 6. **Correlation Analysis**

```python
# Correlation matrix
correlation_matrix = df[numeric_cols].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

print(pd.DataFrame(high_corr_pairs))
```

### 7. **Relationship Analysis**

#### Scatter Plots
```python
# Scatter plot untuk pairs of variables
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Example scatter plots
sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=axes[0, 0])
sns.scatterplot(data=df, x=numeric_cols[2], y=numeric_cols[3], ax=axes[0, 1])

# Additional relationships
# [customize based on dataset]

plt.tight_layout()
plt.show()
```

#### Time Series Analysis (if time column exists)
```python
# Assuming there's a date column
# df['date'] = pd.to_datetime(df['date_column'])

# Time series plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['date'], df['cases'], label='Cases')
ax.plot(df['date'], df['deaths'], label='Deaths')
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('COVID-19 Cases and Deaths Over Time')
ax.legend()
plt.tight_layout()
plt.show()
```

### 8. **Outlier Detection**

```python
# Method 1: IQR method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")

# Method 2: Z-score method
from scipy.stats import zscore

for col in numeric_cols:
    z_scores = np.abs(zscore(df[col].dropna()))
    outliers = (z_scores > 3).sum()
    print(f"{col}: {outliers} outliers (|z-score| > 3)")
```

### 9. **Comparative Analysis**

#### Group By Analysis
```python
# If there are categorical grouping variables
if len(categorical_cols) > 0:
    # Group statistics
    grouped = df.groupby(categorical_cols[0])[numeric_cols].agg(['mean', 'std', 'count'])
    print(grouped)
    
    # Visualize
    df.boxplot(column=numeric_cols[0], by=categorical_cols[0])
    plt.show()
```

### 10. **Key Insights & Findings**

```python
# Summarize key findings
findings = []

findings.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
findings.append(f"Data types: {df.dtypes.value_counts().to_dict()}")
findings.append(f"Missing data: {df.isnull().sum().sum()} total missing values")

print("KEY FINDINGS:")
for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")
```

## 📈 Typical EDA Outputs

- **Data Summary Table** - Shape, types, missing values
- **Distribution Plots** - Histograms, density plots
- **Box Plots** - Outliers & quartiles
- **Correlation Heatmap** - Feature relationships
- **Scatter Plots** - Bivariate relationships
- **Count Plots** - Categorical distributions
- **Time Series Plots** - Temporal patterns (if applicable)
- **Outlier Analysis** - Detection & reporting
- **Summary Statistics** - Mean, std, min, max, etc.
- **Key Insights Document** - Main findings

## 💡 EDA Best Practices

### 1. **Systematic Approach**
- Follow structured process: Load → Inspect → Clean → Analyze → Visualize
- Document findings at setiap step
- Ask questions tentang data

### 2. **Data Quality First**
- Always check missing values
- Detect duplicates
- Validate data types
- Identify outliers

### 3. **Effective Visualization**
- Use appropriate plot types
- Label axes clearly
- Use color effectively
- Include titles & legends
- Avoid chart junk

### 4. **Statistical Rigor**
- Calculate relevant statistics
- Use formal tests jika appropriate
- Understand distributions
- Check assumptions

### 5. **Documentation**
- Comment code thoroughly
- Explain findings
- Provide context
- Link insights ke data

## ⚠️ Common EDA Mistakes

| Mistake | Avoid By |
|---------|----------|
| Skipping missing value analysis | Always check df.isnull().sum() first |
| Assuming normal distribution | Always plot distributions |
| Ignoring outliers | Use box plots & IQR method |
| Over-cluttering plots | Keep visualizations simple & clear |
| No data quality checks | Validate before analysis |
| Missing value labels | Always label axes & add legends |

## 💻 Complete EDA Checklist Template

```python
# Complete EDA Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_excel('synthetic_covid19_data.xlsx')

# 2. Inspect
print(df.shape, df.info(), df.describe())

# 3. Missing Values
missing = df.isnull().sum()
print(f"Missing: {missing[missing > 0]}")

# 4. Duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# 5. Data Types
numeric = df.select_dtypes(np.number).columns
categorical = df.select_dtypes('object').columns

# 6. Univariate Analysis
# Distributions, statistics untuk each column

# 7. Bivariate Analysis
# Relationships between pairs of variables

# 8. Correlation
correlation_matrix = df[numeric].corr()
sns.heatmap(correlation_matrix, annot=True)

# 9. Outliers
# IQR method, Z-score method

# 10. Key Findings
# Document major insights & patterns

# 11. Recommendations
# Suggest next steps untuk analysis
```

## 📚 EDA Resources & References

- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Visualization](https://seaborn.pydata.org/)
- [Matplotlib Guide](https://matplotlib.org/)
- [EDA Best Practices](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [Statistical Methods](https://en.wikipedia.org/wiki/Descriptive_statistics)

## ✅ Checklist

- [ ] Load dataset successfully
- [ ] Inspect shape & structure
- [ ] Check data types
- [ ] Analyze missing values
- [ ] Check for duplicates
- [ ] Generate descriptive statistics
- [ ] Create distribution plots
- [ ] Generate correlation matrix
- [ ] Create scatter/relationship plots
- [ ] Detect outliers
- [ ] Group by analysis (if applicable)
- [ ] Document key findings
- [ ] Create visualization summary
- [ ] Provide recommendations

---

**Author:** Kelompok 6  
**Date:** December 2025  
**Status:** ✓ Complete

## 📝 Project Notes

- Dataset: Synthetic COVID-19 data untuk learning purpose
- Focus: Understanding EDA techniques & best practices
- Output: Comprehensive data exploration report
- Next: Use insights untuk predictive modeling

## 🔍 EDA Process Flow

```
Raw Data
   ↓
Load & Inspect
   ↓
Data Quality Check
   ↓
Univariate Analysis
   ↓
Bivariate Analysis
   ↓
Correlation Analysis
   ↓
Outlier Detection
   ↓
Key Insights
   ↓
Recommendations
```

Semoga dokumentasi ini membantu project Kelompok 6! 📊
