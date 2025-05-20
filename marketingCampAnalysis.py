import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, 
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from lightgbm import LGBMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_marketing_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path, delimiter=';'):
    """
    Load the dataset with error handling and additional diagnostics
    """
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic data overview
        logger.info(f"Column names: {df.columns.tolist()}")
        logger.info(f"Data types: \n{df.dtypes}")
        
        # Check for encoding issues
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col].str.encode('utf-8')
            except Exception as e:
                logger.warning(f"Encoding issue detected in column {col}: {str(e)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def validate_data(df):
    """
    Perform comprehensive data validation
    """
    # Check for required columns
    required_cols = ['ID', 'Year_Birth', 'Income', 'Dt_Customer', 'Response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if 'Year_Birth' in df.columns and not pd.api.types.is_numeric_dtype(df['Year_Birth']):
        logger.warning("Year_Birth column is not numeric")
    
    # Check for duplicate IDs
    duplicates = df.duplicated('ID').sum() if 'ID' in df.columns else 0
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate IDs")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_pct
    })
    logger.info(f"Missing values analysis:\n{missing_info[missing_info['Missing Values'] > 0]}")
    
    # Check for outliers in numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            logger.info(f"Column {col} has {outliers} outliers ({outliers/len(df):.2%})")
    
    # Check for unexpected values
    if 'Year_Birth' in df.columns:
        unrealistic_birth = ((df['Year_Birth'] < 1900) | (df['Year_Birth'] > 2005)).sum()
        if unrealistic_birth > 0:
            logger.warning(f"Found {unrealistic_birth} unrealistic birth years")
    
    # Check class balance for target variable
    if 'Response' in df.columns:
        response_counts = df['Response'].value_counts()
        logger.info(f"Target variable distribution:\n{response_counts}")
        
        minority_pct = response_counts.min() / response_counts.sum() * 100
        if minority_pct < 10:
            logger.warning(f"Highly imbalanced target variable: minority class is only {minority_pct:.2f}%")
    
    return df

def clean_data(df):
    """
    Enhanced data cleaning with outlier handling and improved missing value imputation
    """
    logger.info("Starting enhanced data cleaning process")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Fill missing Income values with median by education level if available
    if 'Income' in df_clean.columns and 'Education' in df_clean.columns:
        missing_income = df_clean['Income'].isnull().sum()
        if missing_income > 0:
            logger.info(f"Filling {missing_income} missing Income values with median by Education group")
            income_by_education = df_clean.groupby('Education')['Income'].transform('median')
            df_clean['Income'].fillna(income_by_education, inplace=True)
            # If still missing, use overall median
            still_missing = df_clean['Income'].isnull().sum()
            if still_missing > 0:
                logger.info(f"Filling remaining {still_missing} Income values with overall median")
                df_clean['Income'].fillna(df_clean['Income'].median(), inplace=True)
    elif 'Income' in df_clean.columns:
        missing_income = df_clean['Income'].isnull().sum()
        if missing_income > 0:
            logger.info(f"Filling {missing_income} missing Income values with median")
            df_clean['Income'].fillna(df_clean['Income'].median(), inplace=True)
    
    # Handle outliers in Income
    if 'Income' in df_clean.columns:
        q1 = df_clean['Income'].quantile(0.25)
        q3 = df_clean['Income'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (3 * iqr)  # Less strict bound
        
        outliers_count = (df_clean['Income'] > upper_bound).sum()
        if outliers_count > 0:
            logger.info(f"Capping {outliers_count} Income outliers")
            df_clean.loc[df_clean['Income'] > upper_bound, 'Income'] = upper_bound
    
    # Remove constant columns
    constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Dropping constant columns: {constant_cols}")
        df_clean.drop(columns=constant_cols, inplace=True)
    
    # Try to drop Z_CostContact and Z_Revenue if they exist
    for col in ['Z_CostContact', 'Z_Revenue']:
        if col in df_clean.columns:
            df_clean.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column: {col}")
    
    # Convert Dt_Customer to datetime
    if 'Dt_Customer' in df_clean.columns:
        try:
            df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], errors='coerce')
            # Handle any failed conversions
            invalid_dates = df_clean['Dt_Customer'].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in Dt_Customer")
                # Use median date to fill missing values
                median_date = df_clean['Dt_Customer'].median()
                df_clean['Dt_Customer'].fillna(median_date, inplace=True)
            logger.info("Converted Dt_Customer to datetime format")
        except Exception as e:
            logger.warning(f"Could not convert Dt_Customer to datetime: {str(e)}")
    
    # Check for and remove duplicate rows
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Removing {duplicates} duplicate rows")
        df_clean = df_clean.drop_duplicates()
    
    logger.info("Enhanced data cleaning completed")
    return df_clean

def create_features(df):
    """
    Create comprehensive features including time-based and RFM features
    """
    logger.info("Starting advanced feature engineering")
    
    df_feat = df.copy()
    
    # Calculate customer tenure and time-based features
    if 'Dt_Customer' in df_feat.columns and pd.api.types.is_datetime64_dtype(df_feat['Dt_Customer']):
        # Use end of 2014 as reference date (assuming data is from 2014 or earlier)
        reference_date = pd.to_datetime("2014-12-31")
        
        # Calculate tenure in days
        df_feat['Customer_Tenure_Days'] = (reference_date - df_feat['Dt_Customer']).dt.days
        
        # Create tenure bands
        df_feat['Tenure_Segment'] = pd.cut(
            df_feat['Customer_Tenure_Days'],
            bins=[0, 90, 365, 730, float('inf')],
            labels=['New (<3mo)', 'Developing (3-12mo)', 'Established (1-2yr)', 'Loyal (>2yr)']
        )
        
        # Extract year and month for seasonality analysis
        df_feat['Enrollment_Year'] = df_feat['Dt_Customer'].dt.year
        df_feat['Enrollment_Month'] = df_feat['Dt_Customer'].dt.month
        df_feat['Enrollment_Quarter'] = df_feat['Dt_Customer'].dt.quarter
        
        # Create cohort identifier (YearMonth)
        df_feat['Cohort'] = df_feat['Dt_Customer'].dt.to_period('M')
        
        logger.info("Created time-based features including tenure and enrollment patterns")
    
    # Create spending features
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    
    # Check if spending columns exist before creating TotalSpend
    existing_spend_cols = [col for col in spending_cols if col in df_feat.columns]
    if existing_spend_cols:
        # Total spending
        df_feat['TotalSpend'] = df_feat[existing_spend_cols].sum(axis=1)
        
        # Calculate spending category proportions
        for col in existing_spend_cols:
            prop_col = f"{col}_Proportion"
            df_feat[prop_col] = df_feat[col] / df_feat['TotalSpend'].replace(0, np.nan)
            df_feat[prop_col].fillna(0, inplace=True)
        
        # Spending diversity (number of categories with non-zero spend)
        df_feat['SpendingDiversity'] = (df_feat[existing_spend_cols] > 0).sum(axis=1)
        
        # Identify primary spending category
        category_cols = {
            'MntWines': 'Wines',
            'MntFruits': 'Fruits',
            'MntMeatProducts': 'Meat',
            'MntFishProducts': 'Fish',
            'MntSweetProducts': 'Sweets',
            'MntGoldProds': 'Gold'
        }
        
        available_categories = {col: name for col, name in category_cols.items() if col in existing_spend_cols}
        if available_categories:
            df_feat['PrimarySpendCategory'] = df_feat[list(available_categories.keys())].idxmax(axis=1)
            # Replace column names with category names
            df_feat['PrimarySpendCategory'] = df_feat['PrimarySpendCategory'].map(category_cols)
        
        logger.info(f"Created spending features including proportions and diversity metrics")
    
    # Create campaign response features
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                    'AcceptedCmp4', 'AcceptedCmp5']
    
    existing_campaign_cols = [col for col in campaign_cols if col in df_feat.columns]
    if existing_campaign_cols:
        # Total campaign acceptances
        df_feat['TotalAcceptedCampaigns'] = df_feat[existing_campaign_cols].sum(axis=1)
        
        # Campaign response rate
        df_feat['CampaignResponseRate'] = df_feat['TotalAcceptedCampaigns'] / len(existing_campaign_cols)
        
        logger.info("Created campaign response features")
    
    # Create Children feature
    if 'Kidhome' in df_feat.columns and 'Teenhome' in df_feat.columns:
        df_feat['Children'] = df_feat['Kidhome'] + df_feat['Teenhome']
        df_feat['HasChildren'] = (df_feat['Children'] > 0).astype(int)
        logger.info("Created Children features")
    
    # Calculate Age and related features
    if 'Year_Birth' in df_feat.columns:
        current_year = 2014  # Using 2014 as reference year
        df_feat['Age'] = current_year - df_feat['Year_Birth']
        logger.info("Created Age feature")
        
        # Create generational segments
        df_feat['Generation'] = pd.cut(
            df_feat['Age'],
            bins=[0, 24, 40, 56, 75, 100],
            labels=['Gen Z', 'Millennials', 'Gen X', 'Boomers', 'Silent']
        )
        
        # Create life stage based on age and children
        if 'HasChildren' in df_feat.columns:
            conditions = [
                (df_feat['Age'] < 35) & (df_feat['HasChildren'] == 0),
                (df_feat['Age'] < 35) & (df_feat['HasChildren'] > 0),
                (df_feat['Age'] >= 35) & (df_feat['Age'] < 50) & (df_feat['HasChildren'] == 0),
                (df_feat['Age'] >= 35) & (df_feat['Age'] < 50) & (df_feat['HasChildren'] > 0),
                (df_feat['Age'] >= 50) & (df_feat['HasChildren'] == 0),
                (df_feat['Age'] >= 50) & (df_feat['HasChildren'] > 0)
            ]
            choices = [
                'Young Single/Couple', 
                'Young Family', 
                'Middle-Aged Single/Couple', 
                'Middle-Aged Family', 
                'Older Single/Couple', 
                'Older Family'
            ]
            df_feat['LifeStage'] = np.select(conditions, choices, default='Unknown')
        
        logger.info("Created generational and life stage features")
    
    # Create RFM (Recency, Frequency, Monetary) features
    if 'Dt_Customer' in df_feat.columns and 'TotalSpend' in df_feat.columns:
        # Recency (already calculated as Customer_Tenure_Days)
        # For frequency, we'll use number of purchases
        purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        existing_purchase_cols = [col for col in purchase_cols if col in df_feat.columns]
        
        if existing_purchase_cols:
            # Frequency
            df_feat['TotalPurchases'] = df_feat[existing_purchase_cols].sum(axis=1)
            
            # Create channel preference
            channel_mapping = {
                'NumDealsPurchases': 'Deals',
                'NumWebPurchases': 'Web',
                'NumCatalogPurchases': 'Catalog',
                'NumStorePurchases': 'Store'
            }
            
            available_channels = {col: name for col, name in channel_mapping.items() if col in existing_purchase_cols}
            if available_channels:
                df_feat['PreferredChannel'] = df_feat[list(available_channels.keys())].idxmax(axis=1)
                df_feat['PreferredChannel'] = df_feat['PreferredChannel'].map(channel_mapping)
                
                # Channel diversity (number of channels used)
                df_feat['ChannelDiversity'] = (df_feat[list(available_channels.keys())] > 0).sum(axis=1)
            
            # Monetary (already calculated as TotalSpend)
            
            # Average spend per purchase
            df_feat['AvgSpend'] = df_feat['TotalSpend'] / df_feat['TotalPurchases'].replace(0, np.nan)
            df_feat['AvgSpend'].fillna(0, inplace=True)
            
            # RFM Score components
            df_feat['R_Score'] = pd.qcut(df_feat['Customer_Tenure_Days'].rank(method='first'), 
                                       q=5, labels=[5, 4, 3, 2, 1])
            df_feat['F_Score'] = pd.qcut(df_feat['TotalPurchases'].rank(method='first'), 
                                       q=5, labels=[1, 2, 3, 4, 5])
            df_feat['M_Score'] = pd.qcut(df_feat['TotalSpend'].rank(method='first'), 
                                       q=5, labels=[1, 2, 3, 4, 5])
            
            # Combined RFM Score
            df_feat['RFM_Score'] = df_feat['R_Score'].astype(str) + df_feat['F_Score'].astype(str) + df_feat['M_Score'].astype(str)
            
            # RFM Segments
            def rfm_segment(row):
                r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
                if r >= 4 and f >= 4 and m >= 4:
                    return 'Champions'
                elif r >= 3 and f >= 3 and m >= 3:
                    return 'Loyal Customers'
                elif r >= 3 and f >= 1 and m >= 2:
                    return 'Potential Loyalists'
                elif r >= 4 and f >= 1 and m >= 1:
                    return 'New Customers'
                elif r <= 2 and f >= 3 and m >= 3:
                    return 'At Risk'
                elif r <= 2 and f >= 2 and m >= 2:
                    return 'Needs Attention'
                elif r <= 1 and f >= 2 and m >= 2:
                    return 'About To Leave'
                elif r <= 2 and f <= 2 and m <= 2:
                    return 'Dormant'
                else:
                    return 'Others'
                    
            df_feat['RFM_Segment'] = df_feat.apply(rfm_segment, axis=1)
            
            logger.info("Created RFM features and segmentation")
    
    # Create interaction features
    if 'Income' in df_feat.columns and 'Age' in df_feat.columns:
        df_feat['Income_per_Age'] = df_feat['Income'] / df_feat['Age']
        
    if 'TotalSpend' in df_feat.columns and 'Income' in df_feat.columns:
        df_feat['SpendToIncome'] = df_feat['TotalSpend'] / df_feat['Income'].replace(0, np.nan)
        df_feat['SpendToIncome'].fillna(0, inplace=True)
        
    if 'TotalAcceptedCampaigns' in df_feat.columns and 'TotalPurchases' in df_feat.columns:
        df_feat['CampaignToPurchase'] = df_feat['TotalAcceptedCampaigns'] / df_feat['TotalPurchases'].replace(0, np.nan)
        df_feat['CampaignToPurchase'].fillna(0, inplace=True)
    
    logger.info("Created interaction features")
    
    # Calculate CLV (Customer Lifetime Value) estimate
    if 'TotalSpend' in df_feat.columns and 'Customer_Tenure_Days' in df_feat.columns:
        # Simple CLV = (Total Spend / Tenure in years) * Expected future years
        df_feat['Spend_per_Year'] = df_feat['TotalSpend'] / (df_feat['Customer_Tenure_Days'] / 365.25)
        
        # Assuming 3 more years of customer relationship
        df_feat['Estimated_CLV'] = df_feat['Spend_per_Year'] * 3
        
        logger.info("Created Customer Lifetime Value estimates")
    
    # Drop redundant columns if they exist
    redundant_cols = ['Year_Birth']
    drop_cols = [col for col in redundant_cols if col in df_feat.columns]
    if drop_cols:
        df_feat.drop(columns=drop_cols, inplace=True)
        logger.info(f"Dropped redundant columns: {drop_cols}")
    
    logger.info("Advanced feature engineering completed")
    return df_feat

def prepare_for_modeling(df, target_col='Response'):
    """
    Prepare the data for machine learning models with enhanced preprocessing
    """
    logger.info("Preparing data for modeling with enhanced techniques")
    
    df_model = df.copy()
    
    # Ensure we have the target column
    if target_col not in df_model.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe")
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Convert date columns to features and drop
    date_cols = df_model.select_dtypes(include=['datetime64']).columns.tolist()
    for date_col in date_cols:
        logger.info(f"Dropping datetime column: {date_col} (already extracted features)")
        df_model.drop(columns=[date_col], inplace=True)
    
    # Handle period datatypes (like Cohort)
    period_cols = [col for col in df_model.columns if pd.api.types.is_period_dtype(df_model[col])]
    for col in period_cols:
        logger.info(f"Converting period column {col} to string")
        df_model[col] = df_model[col].astype(str)
    
    # Convert categorical variables to dummy variables
    cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column if it's in cat_cols
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Keep ID but don't encode it
    id_col = 'ID' if 'ID' in df_model.columns else None
    if id_col in cat_cols:
        cat_cols.remove(id_col)
    
    # Check number of unique values - if there are categorical columns with many values, consider other encoding
    high_cardinality = []
    for col in cat_cols:
        if df_model[col].nunique() > 10:
            high_cardinality.append((col, df_model[col].nunique()))
    
    if high_cardinality:
        logger.warning(f"High cardinality categorical features detected: {high_cardinality}")
        logger.info("Consider target encoding or other methods for these features")
    
    # One-hot encode remaining categorical columns
    if cat_cols:
        logger.info(f"Converting categorical columns to dummies: {cat_cols}")
        df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    
    # Transform skewed numerical features
    num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove ID and target column from transformation
    transform_cols = [col for col in num_cols if col not in [id_col, target_col]]
    
    # Check for skewness and apply transformation if needed
    skewed_features = []
    for col in transform_cols:
        skewness = df_model[col].skew()
        if abs(skewness) > 1:
            skewed_features.append((col, skewness))
    
    if skewed_features:
        logger.info(f"Found {len(skewed_features)} skewed features to transform")
        
        # Apply Power Transformer to handle skewness
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df_model[transform_cols] = pt.fit_transform(df_model[transform_cols])
    
    # Standard scaling for numerical features
    scale_cols = transform_cols
    
    if scale_cols:
        logger.info(f"Scaling {len(scale_cols)} numerical columns")
        scaler = StandardScaler()
        df_model[scale_cols] = scaler.fit_transform(df_model[scale_cols])
    
    # Check for multicollinearity
    try:
        logger.info("Checking for multicollinearity")
        X = df_model.drop(columns=[col for col in [id_col, target_col] if col in df_model.columns])
        
        # Sample data if too large (VIF calculation can be slow)
        if X.shape[0] > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X
            
        # Get a subset of features (max 30) for VIF calculation to avoid performance issues
        if X_sample.shape[1] > 30:
            logger.info("Selecting top 30 features for VIF calculation")
            # Simple correlation-based selection
            corr_with_target = abs(df_model.drop([id_col], axis=1).corrwith(df_model[target_col]))
            top_features = corr_with_target.sort_values(ascending=False).head(30).index.tolist()
            vif_cols = [col for col in top_features if col != target_col and col in X_sample.columns]
        else:
            vif_cols = X_sample.columns.tolist()
            
        vif_data = pd.DataFrame()
        vif_data["Feature"] = vif_cols
        vif_data["VIF"] = [variance_inflation_factor(X_sample[vif_cols].values, i) for i in range(len(vif_cols))]
        
        high_vif = vif_data[vif_data["VIF"] > 10]
        if not high_vif.empty:
            logger.warning(f"High multicollinearity detected in features: \n{high_vif}")
    except Exception as e:
        logger.warning(f"Couldn't calculate VIF: {str(e)}")
    
    logger.info("Enhanced data preparation completed")
    return df_model

def train_classification_model(df, target_col='Response'):
    """
    Train and evaluate multiple classification models with advanced techniques
    """
    logger.info("Training classification models with advanced techniques")
    
    # Define features and target
    id_col = 'ID' if 'ID' in df.columns else None
    drop_cols = [col for col in [id_col, target_col] if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Check class distribution and handle imbalance if necessary
    class_counts = y.value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    minority_pct = class_counts.min() / class_counts.sum() * 100
    imbalanced = minority_pct < 15
    
    if imbalanced:
        logger.warning(f"Imbalanced classes detected: minority class is {minority_pct:.2f}%")
        class_weight = 'balanced'
        scoring_metric = 'f1'
        logger.info("Using balanced class weights and F1 score for optimization")
    else:
        class_weight = None
        scoring_metric = 'roc_auc'
        logger.info("Using ROC AUC for optimization")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weight),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', 
                                    scale_pos_weight=None if not imbalanced else class_counts[0]/class_counts[1]),
        'LightGBM': LGBMClassifier(random_state=42, class_weight=class_weight)
    }
    
    # Model tuning parameters
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
    
    # Store results
    model_results = {}
    best_models = {}
    
    # Feature selection through model-based selection
    logger.info("Performing feature selection using Random Forest")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
    rf_selector.fit(X_train, y_train)
    
    # Get feature importances and select top features
    feature_importances = pd.DataFrame(
        {'feature': X_train.columns, 'importance': rf_selector.feature_importances_}
    ).sort_values('importance', ascending=False)
    
    # Select features that account for 95% of cumulative importance
    feature_importances['cumulative_importance'] = feature_importances['importance'].cumsum()
    important_features = feature_importances[feature_importances['cumulative_importance'] <= 0.95]['feature'].tolist()
    
    logger.info(f"Selected {len(important_features)} out of {X_train.shape[1]} features based on importance")
    logger.info(f"Top 10 features: {', '.join(feature_importances['feature'].head(10).tolist())}")
    
    # Use these features for all models
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]
    
    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name} with GridSearchCV")
        
        # Get parameter grid for the current model
        param_grid = param_grids[model_name]
        
        # Setup grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit model
        try:
            grid_search.fit(X_train_selected, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model
            
            # Make predictions
            y_pred = best_model.predict(X_test_selected)
            y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
            
            # Evaluate
            accuracy = best_model.score(X_test_selected, y_test)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # ROC AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Store metrics
            model_results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'best_params': grid_search.best_params_,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
            
            logger.info(f"{model_name} results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  ROC AUC: {roc_auc:.4f}")
            logger.info(f"  PR AUC: {pr_auc:.4f}")
            logger.info(f"  Best params: {grid_search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
    
    # Find best model based on selected scoring metric
    best_score = -1
    best_model_name = None
    
    for model_name, results in model_results.items():
        score = results['roc_auc'] if scoring_metric == 'roc_auc' else results['f1_score']
        if score > best_score:
            best_score = score
            best_model_name = model_name
    
    if best_model_name:
        logger.info(f"Best model: {best_model_name} with {scoring_metric}={best_score:.4f}")
    else:
        logger.warning("No models were successfully trained")
    
    # SHAP feature importance for best model
    if best_model_name and best_model_name in best_models:
        try:
            logger.info(f"Calculating SHAP values for {best_model_name}")
            best_model = best_models[best_model_name]
            
            if best_model_name == 'RandomForest':
                explainer = shap.TreeExplainer(best_model)
            elif best_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(best_model)
            else:
                explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train_selected, 100))
            
            # Sample for SHAP explanation if dataset is large
            if X_test_selected.shape[0] > 100:
                X_shap = X_test_selected.sample(100, random_state=42)
            else:
                X_shap = X_test_selected
            
            shap_values = explainer.shap_values(X_shap)
            
            # For tree-based models, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use class 1
                
            # Get feature importance based on SHAP
            shap_importance = pd.DataFrame({
                'feature': X_train_selected.columns,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            logger.info(f"Top 10 features by SHAP importance: {', '.join(shap_importance['feature'].head(10).tolist())}")
            
        except Exception as e:
            logger.warning(f"Error calculating SHAP values: {str(e)}")
    
    return best_models, model_results, important_features

def visualize_results(df, model_results, important_features):
    """
    Create visualizations of model results and data insights
    """
    logger.info("Creating visualizations of results")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = "marketing_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Feature Importance Plot
        if important_features:
            plt.figure(figsize=(12, 8))
            
            # Get top 15 features
            top_features = important_features[:min(15, len(important_features))]
            feature_importance = pd.DataFrame({
                'feature': top_features,
                'importance': range(len(top_features), 0, -1)
            }).sort_values('importance')
            
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Top 15 Features by Importance')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png")
            plt.close()
            logger.info(f"Saved feature importance plot to {output_dir}/feature_importance.png")
        
        # 2. Model Performance Comparison
        if model_results:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'pr_auc']
            model_metrics = []
            
            for model_name, results in model_results.items():
                model_data = {'model': model_name}
                for metric in metrics:
                    if metric in results:
                        model_data[metric] = results[metric]
                model_metrics.append(model_data)
            
            model_metrics_df = pd.DataFrame(model_metrics)
            
            plt.figure(figsize=(14, 10))
            sns.set_theme(style="whitegrid")
            
            # Reshape for better plotting
            model_metrics_long = pd.melt(model_metrics_df, id_vars=['model'], 
                                         value_vars=metrics, 
                                         var_name='metric', value_name='score')
            
            sns.barplot(x='model', y='score', hue='metric', data=model_metrics_long)
            plt.title('Model Performance Comparison')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_comparison.png")
            plt.close()
            logger.info(f"Saved model comparison plot to {output_dir}/model_comparison.png")
            
            # 3. ROC Curves for all models
            plt.figure(figsize=(10, 8))
            
            for model_name, results in model_results.items():
                if 'y_pred_proba' in results and len(results['y_pred_proba']) > 0:
                    # Get test data and predictions
                    y_true = df['Response'] if 'Response' in df.columns else None
                    y_score = results['y_pred_proba']
                    
                    if y_true is not None and len(y_true) >= len(y_score):
                        # Use the same test indices
                        y_true = y_true.iloc[-len(y_score):]
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = results.get('roc_auc', auc(fpr, tpr))
                        
                        # Plot ROC curve
                        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/roc_curves.png")
            plt.close()
            logger.info(f"Saved ROC curves to {output_dir}/roc_curves.png")
        
        # 4. Customer Segmentation Visualization (if RFM segments exist)
        if 'RFM_Segment' in df.columns:
            segment_counts = df['RFM_Segment'].value_counts()
            
            plt.figure(figsize=(12, 8))
            segment_counts.plot(kind='bar', color=sns.color_palette("viridis", len(segment_counts)))
            plt.title('Customer Segments Distribution')
            plt.xlabel('Segment')
            plt.ylabel('Number of Customers')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/customer_segments.png")
            plt.close()
            logger.info(f"Saved customer segments plot to {output_dir}/customer_segments.png")
            
            # Segment profiles
            segment_profiles = df.groupby('RFM_Segment').agg({
                'Age': 'mean',
                'Income': 'mean',
                'TotalSpend': 'mean',
                'TotalPurchases': 'mean',
                'TotalAcceptedCampaigns': 'mean',
                'Customer_Tenure_Days': 'mean',
                'Response': 'mean'
            }).reset_index()
            
            segment_profiles['Response_Rate'] = segment_profiles['Response'] * 100
            segment_profiles['Tenure_Years'] = segment_profiles['Customer_Tenure_Days'] / 365.25
            
            # Save segment profiles
            segment_profiles.to_csv(f"{output_dir}/segment_profiles.csv", index=False)
            logger.info(f"Saved segment profiles to {output_dir}/segment_profiles.csv")
        
        # 5. Campaign Response Analysis
        campaign_cols = [col for col in df.columns if col.startswith('Accepted')]
        if campaign_cols:
            campaign_response = df[campaign_cols].mean().reset_index()
            campaign_response.columns = ['Campaign', 'Response_Rate']
            campaign_response['Response_Rate'] = campaign_response['Response_Rate'] * 100
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Campaign', y='Response_Rate', data=campaign_response)
            plt.title('Campaign Response Rates')
            plt.xlabel('Campaign')
            plt.ylabel('Response Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/campaign_response_rates.png")
            plt.close()
            logger.info(f"Saved campaign response rates plot to {output_dir}/campaign_response_rates.png")
            
        # 6. Interactive visualizations with Plotly
        try:
            # Customer Spend by Age Group
            if 'Age' in df.columns and 'TotalSpend' in df.columns:
                df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                                      labels=['<30', '30-40', '40-50', '50-60', '60+'])
                
                age_spend = df.groupby('AgeGroup').agg({
                    'TotalSpend': 'mean',
                    'ID': 'count'
                }).reset_index()
                
                age_spend['CustomerCount'] = age_spend['ID']
                age_spend.drop('ID', axis=1, inplace=True)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(x=age_spend['AgeGroup'], y=age_spend['TotalSpend'], name='Avg. Spend'),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=age_spend['AgeGroup'], y=age_spend['CustomerCount'], 
                              mode='lines+markers', name='Customer Count'),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='Average Spend and Customer Count by Age Group',
                    xaxis_title='Age Group',
                    yaxis_title='Average Spend',
                    yaxis2_title='Customer Count'
                )
                
                fig.write_html(f"{output_dir}/age_spend_analysis.html")
                logger.info(f"Saved interactive age spend analysis to {output_dir}/age_spend_analysis.html")
            
            # Response Rate by Customer Segments
            if 'RFM_Segment' in df.columns and 'Response' in df.columns:
                segment_response = df.groupby('RFM_Segment').agg({
                    'Response': 'mean',
                    'ID': 'count'
                }).reset_index()
                
                segment_response['ResponseRate'] = segment_response['Response'] * 100
                segment_response['CustomerCount'] = segment_response['ID']
                segment_response.drop('ID', axis=1, inplace=True)
                
                # Sort by response rate
                segment_response = segment_response.sort_values('ResponseRate', ascending=False)
                
                fig = px.bar(segment_response, x='RFM_Segment', y='ResponseRate',
                            text='ResponseRate', color='CustomerCount',
                            labels={'ResponseRate': 'Response Rate (%)', 
                                   'RFM_Segment': 'Customer Segment',
                                   'CustomerCount': 'Customer Count'},
                            title='Response Rate by Customer Segment')
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                fig.write_html(f"{output_dir}/segment_response_analysis.html")
                logger.info(f"Saved interactive segment response analysis to {output_dir}/segment_response_analysis.html")
                
        except Exception as e:
            logger.warning(f"Error creating interactive Plotly visualizations: {str(e)}")
            
        logger.info("All visualizations created successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")

def save_results(best_models, model_results, important_features, df):
    """
    Save models, results and processed data
    """
    logger.info("Saving models, results and processed data")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = "marketing_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best models
        for model_name, model in best_models.items():
            with open(f"{output_dir}/{model_name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} model to {output_dir}/{model_name}_model.pkl")
        
        # Save model results
        with open(f"{output_dir}/model_results.pkl", 'wb') as f:
            pickle.dump(model_results, f)
        logger.info(f"Saved model results to {output_dir}/model_results.pkl")
        
        # Save important features
        with open(f"{output_dir}/important_features.pkl", 'wb') as f:
            pickle.dump(important_features, f)
        logger.info(f"Saved important features to {output_dir}/important_features.pkl")
        
        # Save processed dataframe
        df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        logger.info(f"Saved processed data to {output_dir}/processed_data.csv")
        
        # Save feature documentation
        feature_docs = []
        for col in df.columns:
            if col in ['ID', 'Response']:
                continue
                
            # Determine feature type
            if pd.api.types.is_numeric_dtype(df[col]):
                feat_type = "Numeric"
                feat_stats = f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean: {df[col].mean():.2f}"
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                feat_type = "Categorical"
                feat_stats = f"Unique values: {df[col].nunique()}"
            else:
                feat_type = "Other"
                feat_stats = ""
                
            # Determine if feature is in important features
            if col in important_features:
                importance = "Yes"
                rank = important_features.index(col) + 1
            else:
                importance = "No"
                rank = "-"
                
            feature_docs.append({
                "Feature": col,
                "Type": feat_type,
                "Statistics": feat_stats,
                "Important": importance,
                "Importance Rank": rank
            })
            
        feature_docs_df = pd.DataFrame(feature_docs)
        feature_docs_df.to_csv(f"{output_dir}/feature_documentation.csv", index=False)
        logger.info(f"Saved feature documentation to {output_dir}/feature_documentation.csv")
        
        logger.info("All results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def main(file_path='marketing_data.csv'):
    """
    Main function to execute the full analysis pipeline
    """
    start_time = datetime.now()
    logger.info(f"Starting marketing campaign analysis at {start_time}")
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Validate data
        df = validate_data(df)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Create features
        df_features = create_features(df_clean)
        
        # Prepare for modeling
        df_model = prepare_for_modeling(df_features)
        
        # Train models
        best_models, model_results, important_features = train_classification_model(df_model)
        
        # Visualize results
        visualize_results(df_features, model_results, important_features)
        
        # Save results
        save_results(best_models, model_results, important_features, df_features)
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Analysis completed successfully in {execution_time}")
        
        return {
            'status': 'success',
            'execution_time': str(execution_time),
            'models': list(best_models.keys()),
            'data_shape': df_features.shape
        }
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        return {
            'status': 'error',
            'error_message': str(e)
        }

if __name__ == "__main__":
    # Run with custom file path if provided as command line argument
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "marketing_campaign.csv"
    result = main(file_path)
    print(f"Analysis result: {result}")