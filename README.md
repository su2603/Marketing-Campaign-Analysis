# Marketing Campaign Analysis App Documentation

## Overview

This project provides an end-to-end pipeline for marketing campaign analysis, including data ingestion, preprocessing, feature engineering, customer segmentation, predictive modeling, and interactive visualization via a Streamlit app.

### Components

* **`marketingCampAnalysis.py`**: Backend pipeline for data cleaning, feature generation, and training classification models using Scikit-learn, XGBoost, and LightGBM.
* **`marketingCampaignAnalysisApp.py`**: Streamlit-based UI to explore data, visualize trends, understand customer behavior, and predict marketing responses.

---

## File: `marketingCampAnalysis.py`

### Purpose

Implements backend logic for detailed data validation, cleaning, feature extraction, RFM analysis, modeling, and performance evaluation with logging and interpretability features.

### Key Components

#### 1. **Imports**

* Data: `pandas`, `numpy`
* Visualization: `matplotlib`, `seaborn`, `plotly`
* Modeling: `sklearn`, `xgboost`, `lightgbm`
* Evaluation: `f1_score`, `roc_auc_score`, etc.
* Explainability: `shap`
* Utility: `logging`, `os`, `pickle`

#### 2. **Functions**

##### `load_data(file_path, delimiter=';')`

* Reads the CSV file.
* Detects encoding issues.
* Logs column data types and dimensions.

##### `validate_data(df)`

* Checks for required columns and duplicate IDs.
* Logs missing data stats.
* Flags unrealistic values (e.g., birth year).
* Assesses class imbalance in target.

##### `clean_data(df)`

* Handles missing values intelligently (e.g., median income by education).
* Caps outliers using IQR.
* Converts date columns.
* Drops constant or redundant columns.

##### `create_features(df)`

* **Time-Based**: Customer tenure, enrollment year/month/quarter, cohort.
* **Spending**: Total spend, category proportions, primary category.
* **Campaigns**: Acceptance count, response rate.
* **Demographics**: Age, generation, lifestage.
* **RFM**: Recency (tenure), Frequency (purchases), Monetary (total spend).
* **Channels**: Preferred channel, channel diversity.
* **Derived Metrics**: Spend per year, CLV, income per age, spend-to-income ratio.

##### `prepare_for_modeling(df, target_col='Response')`

* One-hot encodes categoricals.
* Transforms skewed features using PowerTransformer.
* Standardizes numerical features.
* Calculates VIF to flag multicollinearity.

##### `train_classification_model(df, target_col='Response')`

* Supports RandomForest, XGBoost, LightGBM.
* Uses `GridSearchCV` for hyperparameter tuning.
* Selects top features based on cumulative importance.
* Evaluates performance using F1, AUC, PR AUC.
* Calculates SHAP values for top feature explainability.

##### `visualize_results(df, model_results, important_features)`

* Saves feature importance plots and other performance charts to disk.

---

## File: `marketingCampaignAnalysisApp.py`

### Purpose

Frontend interface using Streamlit for user-friendly exploration and interaction with the dataset and models.

### Layout & Features

#### **1. Sidebar Controls**

* File upload and delimiter selection
* Data processing trigger
* Feature creation and model training controls

#### **2. Main Tabs**

##### `Data Overview`

* View sample dataset
* Download processed CSV
* Summary stats
* Histograms for Age, Income
* Correlation matrix heatmap

##### `Customer Segments`

* RFM segmentation pie chart
* Response rate by segment bar chart
* Demographic breakdown (generation, education, lifestage)
* Spend vs. Avg purchase scatter plot

##### `Spending Analysis`

* Category-wise spending (wine, meat, fruits, etc.)
* Diversity and dominant product interest

##### `Campaign Analysis`

* Response rates across campaigns
* Total acceptances, response effectiveness
* Time-based trends (monthly, quarterly)

##### `Response Prediction`

* User input fields (age, education, income, etc.)
* Predicts whether user will respond
* Displays prediction result (0/1) with probabilities

### Visual Tools

* `Plotly` interactive charts
* `Seaborn` and `Matplotlib` for summary plots

---

## Usage Instructions

1. Ensure required packages are installed:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost shap streamlit
   ```

2. Place the dataset (e.g., `marketing_campaign.csv`) in the root folder.

3. Run the Streamlit app:

   ```bash
   streamlit run marketingCampaignAnalysisApp.py
   ```

4. Follow these steps in the app:

   * Upload the dataset.
   * Click "Load Data", then "Clean & Transform".
   * Optionally train model to enable predictions.
   * Use tabs to explore segmentation, spend, and campaign insights.

---

## Output Files

* `rf_model.pkl`: Trained Random Forest model.
* `encoders.pkl`: LabelEncoders for categorical fields.
* `*.png`: Saved charts from `visualize_results()`.
* `enhanced_marketing_analysis.log`: Logs for all preprocessing and training steps.

---

## Notes & Best Practices

* **Data Consistency**: Ensure input schema matches expected column names.
* **Model Updates**: Re-run training whenever data is refreshed.
* **Interpretability**: SHAP values are used to explain model decisions.
* **Segmentation Strategy**: Use RFM and demographic segments together for better targeting.
* **CLV**: Estimated based on spend/year, assumes 3 years of future engagement.

---

##



