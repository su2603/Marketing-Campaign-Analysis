import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import pickle
from datetime import datetime
import base64
import logging
import sys
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# ML libraries (will be imported as needed)
try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, auc, 
        precision_recall_curve, average_precision_score, 
        roc_auc_score, f1_score, precision_score, recall_score
    )
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel
    ML_IMPORTS_SUCCESS = True
except ImportError:
    ML_IMPORTS_SUCCESS = False
    st.warning("Some machine learning libraries couldn't be imported. Model training functionality may be limited.")

# Set page configuration
st.set_page_config(
    page_title="Marketing Campaign Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stProgress .st-eb {
        background-color: #4CAF50;
    }
    .st-emotion-cache-10trblm {
        position: relative;
        text-align: center;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-16txtl3 h4 {
        padding-top: 1rem;
        font-weight: 600;
    }
    .highlight {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ================ Helper Functions ================

def load_data(uploaded_file, delimiter=';'):
    """Load and validate the uploaded CSV file"""
    try:
        # Check if delimiter was automatically detected
        df = pd.read_csv(uploaded_file, sep=delimiter)
        
        # If we only got one column, try to detect the separator
        if df.shape[1] == 1:
            # Try common delimiters
            for delim in [',', '\t', '|']:
                try:
                    df = pd.read_csv(uploaded_file, sep=delim)
                    if df.shape[1] > 1:
                        break
                except:
                    continue
        
        st.success(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def validate_data(df):
    """Perform basic data validation and display stats"""
    
    # Check for required columns
    required_cols = ['Year_Birth', 'Income', 'Response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_cols)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100

    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_pct
    })

    # Add and coerce 'Data Type' column
    missing_info['Data Type'] = df.dtypes.astype(str)

    cols_with_missing = missing_info[missing_info['Missing Values'] > 0]
    if not cols_with_missing.empty:
        st.warning("⚠️ Dataset contains missing values:")
        st.dataframe(cols_with_missing)
    
    # Check for duplicates
    if 'ID' in df.columns:
        duplicates = df.duplicated('ID').sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate IDs")
    
    # Check for class imbalance if Response column exists
    if 'Response' in df.columns:
        response_counts = df['Response'].value_counts()
        
        # Display as percentage
        response_pct = (response_counts / response_counts.sum() * 100).round(2)
        
        # Check for severe imbalance
        minority_pct = response_pct.min()
        if minority_pct < 10:
            st.warning(f"⚠️ Highly imbalanced target variable: minority class is only {minority_pct}%")
    
    return df

def handle_datetime_columns(df, date_columns=None):
    """
    Explicitly handle datetime columns to ensure Arrow compatibility
    
    Args:
        df: DataFrame to process
        date_columns: List of column names to treat as dates (optional)
        
    Returns:
        Processed DataFrame with Arrow-compatible datetime columns
    """
    # Make a copy to avoid altering the original
    df_processed = df.copy()
    
    # If specific date columns are provided
    if date_columns:
        columns_to_process = [col for col in date_columns if col in df_processed.columns]
    else:
        # Try to identify datetime columns
        # First, check for columns with datetime dtype
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Then look for 'date' or 'dt' or 'time' in column names for object types
        potential_date_cols = [
            col for col in df_processed.select_dtypes(include=['object']).columns
            if any(date_term in col.lower() for date_term in ['date', 'dt', 'time'])
        ]
        
        columns_to_process = datetime_cols + potential_date_cols
    
    # If 'Dt_Customer' exists but wasn't detected, add it explicitly
    if 'Dt_Customer' in df_processed.columns and 'Dt_Customer' not in columns_to_process:
        columns_to_process.append('Dt_Customer')
    
    for col in columns_to_process:
        try:
            # Convert to datetime with pandas
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            
            # Fill NaT values with median date
            if df_processed[col].isna().any():
                median_date = df_processed[col].dropna().median()
                df_processed[col] = df_processed[col].fillna(median_date)
            
            # Round to milliseconds (Arrow compatible)
            df_processed[col] = df_processed[col].dt.floor('ms')
            
            # Explicitly convert to datetime64[ms]
            df_processed[col] = df_processed[col].astype('datetime64[ms]')
            
        except Exception as e:
            logger.warning(f"Could not convert {col} to datetime: {str(e)}")
    
    return df_processed

def make_arrow_compatible(df):
    """
    Make all columns in DataFrame compatible with Arrow serialization
    """
    # Create a copy to avoid modifying the original
    df_arrow = df.copy()
    
    # Handle datetime columns
    df_arrow = handle_datetime_columns(df_arrow)
    
    # Handle any period columns by converting to string
    for col in df_arrow.columns:
        if pd.api.types.is_period_dtype(df_arrow[col]):
            df_arrow[col] = df_arrow[col].astype(str)
    
    # Ensure all numeric columns have appropriate types
    for col in df_arrow.select_dtypes(include=['float']).columns:
        df_arrow[col] = df_arrow[col].astype('float64')
    
    for col in df_arrow.select_dtypes(include=['integer']).columns:
        df_arrow[col] = df_arrow[col].astype('int64')
    
    return df_arrow

def clean_data(df):
    """Clean the data and handle missing values"""
    df_clean = df.copy()
    
    # Fill missing Income values with median by education level if possible
    if 'Income' in df_clean.columns and df_clean['Income'].isnull().sum() > 0:
        if 'Education' in df_clean.columns:
            income_by_education = df_clean.groupby('Education')['Income'].transform('median')
            df_clean['Income'].fillna(income_by_education, inplace=True)
        
        # Fill any remaining missing values with overall median
        df_clean['Income'].fillna(df_clean['Income'].median(), inplace=True)
    
    # Handle outliers in Income
    if 'Income' in df_clean.columns:
        q1 = df_clean['Income'].quantile(0.25)
        q3 = df_clean['Income'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (3 * iqr)
        
        outliers_count = (df_clean['Income'] > upper_bound).sum()
        if outliers_count > 0:
            df_clean.loc[df_clean['Income'] > upper_bound, 'Income'] = upper_bound
    
    # Remove constant columns
    constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if constant_cols:
        df_clean.drop(columns=constant_cols, inplace=True)
    
    # Try to drop Z_CostContact and Z_Revenue if they exist
    for col in ['Z_CostContact', 'Z_Revenue']:
        if col in df_clean.columns:
            df_clean.drop(columns=[col], inplace=True)
    
    # Process Dt_Customer datetime column
    df_clean = handle_datetime_columns(df_clean, ['Dt_Customer'])
    
    # Remove duplicates if any
    if df_clean.duplicated().sum() > 0:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean

def create_features(df):
    """Create advanced features for analysis"""
    df_feat = df.copy()
    
    # Calculate customer tenure and time-based features
    if 'Dt_Customer' in df_feat.columns:
        # Make sure it's datetime
        if not pd.api.types.is_datetime64_any_dtype(df_feat['Dt_Customer']):
            df_feat = handle_datetime_columns(df_feat, ['Dt_Customer'])
        
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
        df_feat['Cohort'] = df_feat['Dt_Customer'].dt.to_period('M').astype(str)
    
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
    
    # Create campaign response features
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                    'AcceptedCmp4', 'AcceptedCmp5']
    
    existing_campaign_cols = [col for col in campaign_cols if col in df_feat.columns]
    if existing_campaign_cols:
        # Total campaign acceptances
        df_feat['TotalAcceptedCampaigns'] = df_feat[existing_campaign_cols].sum(axis=1)
        
        # Campaign response rate
        df_feat['CampaignResponseRate'] = df_feat['TotalAcceptedCampaigns'] / len(existing_campaign_cols)
    
    # Create Children feature
    if 'Kidhome' in df_feat.columns and 'Teenhome' in df_feat.columns:
        df_feat['Children'] = df_feat['Kidhome'] + df_feat['Teenhome']
        df_feat['HasChildren'] = (df_feat['Children'] > 0).astype(int)
    
    # Calculate Age and related features
    if 'Year_Birth' in df_feat.columns:
        current_year = 2014  # Using 2014 as reference year
        df_feat['Age'] = current_year - df_feat['Year_Birth']
        
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
    
    # Create RFM (Recency, Frequency, Monetary) features
    if 'Customer_Tenure_Days' in df_feat.columns and 'TotalSpend' in df_feat.columns:
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
            
            # Average spend per purchase
            df_feat['AvgSpend'] = df_feat['TotalSpend'] / df_feat['TotalPurchases'].replace(0, np.nan)
            df_feat['AvgSpend'].fillna(0, inplace=True)
            
            # RFM Score components (using quantiles for more balanced distribution)
            try:
                df_feat['R_Score'] = pd.qcut(df_feat['Customer_Tenure_Days'].rank(method='first'), 
                                           q=5, labels=[5, 4, 3, 2, 1])
                df_feat['F_Score'] = pd.qcut(df_feat['TotalPurchases'].rank(method='first'), 
                                           q=5, labels=[1, 2, 3, 4, 5])
                df_feat['M_Score'] = pd.qcut(df_feat['TotalSpend'].rank(method='first'), 
                                           q=5, labels=[1, 2, 3, 4, 5])
                
                # Convert scores to int
                df_feat['R_Score'] = df_feat['R_Score'].astype(int)
                df_feat['F_Score'] = df_feat['F_Score'].astype(int)
                df_feat['M_Score'] = df_feat['M_Score'].astype(int)
                
                # Combined RFM Score
                df_feat['RFM_Score'] = df_feat['R_Score'].astype(str) + df_feat['F_Score'].astype(str) + df_feat['M_Score'].astype(str)
                
                # RFM Segments
                def rfm_segment(row):
                    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
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
            except Exception as e:
                logger.warning(f"Could not create RFM segments: {str(e)}")
    
    # Create interaction features
    if 'Income' in df_feat.columns and 'Age' in df_feat.columns:
        df_feat['Income_per_Age'] = df_feat['Income'] / df_feat['Age']
        
    if 'TotalSpend' in df_feat.columns and 'Income' in df_feat.columns:
        df_feat['SpendToIncome'] = df_feat['TotalSpend'] / df_feat['Income'].replace(0, np.nan)
        df_feat['SpendToIncome'].fillna(0, inplace=True)
        
    if 'TotalAcceptedCampaigns' in df_feat.columns and 'TotalPurchases' in df_feat.columns:
        df_feat['CampaignToPurchase'] = df_feat['TotalAcceptedCampaigns'] / df_feat['TotalPurchases'].replace(0, np.nan)
        df_feat['CampaignToPurchase'].fillna(0, inplace=True)
    
    # Calculate CLV (Customer Lifetime Value) estimate
    if 'TotalSpend' in df_feat.columns and 'Customer_Tenure_Days' in df_feat.columns:
        # Simple CLV = (Total Spend / Tenure in years) * Expected future years
        df_feat['Spend_per_Year'] = df_feat['TotalSpend'] / (df_feat['Customer_Tenure_Days'] / 365.25)
        df_feat['Spend_per_Year'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Assuming 3 more years of customer relationship
        df_feat['Estimated_CLV'] = df_feat['Spend_per_Year'] * 3
    
    # Drop redundant columns if they exist
    redundant_cols = ['Year_Birth']
    drop_cols = [col for col in redundant_cols if col in df_feat.columns]
    if drop_cols:
        df_feat.drop(columns=drop_cols, inplace=True)
    
    # Make Arrow compatible
    df_feat = make_arrow_compatible(df_feat)
    
    return df_feat

def train_model(df, target_col='Response'):
    """Train a Random Forest model for campaign response prediction"""
    if not ML_IMPORTS_SUCCESS:
        st.error("Required machine learning libraries are not available.")
        return None, None, None
        
    # Define features and target
    id_col = 'ID' if 'ID' in df.columns else None
    drop_cols = [col for col in [id_col, target_col] if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Check class distribution
    class_counts = y.value_counts()
    minority_pct = class_counts.min() / class_counts.sum() * 100
    imbalanced = minority_pct < 15
    
    if imbalanced:
        class_weight = 'balanced'
    else:
        class_weight = None
    
    # Get a list of all datetime columns
    datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Create feature dictionary to store datetime column transformations
    datetime_features = {}

    # Handle datetime columns before train/test split
    for col in datetime_cols:
        # Create standardized feature names
        X[f'{col}_Year'] = X[col].dt.year
        X[f'{col}_Month'] = X[col].dt.month
        X[f'{col}_Day'] = X[col].dt.day
        
        # Store the transformations for later use in prediction
        datetime_features[col] = [f'{col}_Year', f'{col}_Month', f'{col}_Day']

    # Drop the original datetime columns
    X = X.drop(columns=datetime_cols)
    
    # Handle non-numeric columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature selection using Random Forest
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
    rf_selector.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = pd.DataFrame(
        {'feature': X_train.columns, 'importance': rf_selector.feature_importances_}
    ).sort_values('importance', ascending=False)
    
    # Select top features that account for 95% of importance
    feature_importances['cumulative_importance'] = feature_importances['importance'].cumsum()
    important_features = feature_importances[feature_importances['cumulative_importance'] <= 0.95]['feature'].tolist()
    
    # Train model with selected features
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight=class_weight
    )
    
    model.fit(X_train_selected, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Model metrics
    accuracy = model.score(X_test_selected, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist(),
        'datetime_features': datetime_features
    }
    
    return model, results, important_features

def get_download_link(df, filename="processed_data.csv"):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# ================ Main Application ================

def main():
    st.title("📊 Marketing Campaign Analysis")
    
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        This application helps you analyze marketing campaign data to identify customer segments, 
        predict campaign responses, and discover insights that can optimize your marketing strategy.
        
        **Key Features:**
        - 📋 Data exploration and preprocessing
        - 📈 Advanced feature engineering to identify customer patterns
        - 🔍 Customer segmentation using RFM analysis
        - 🎯 Campaign response prediction
        - 📊 Interactive visualizations and insights
        
        **Upload your marketing campaign data to begin!**
        
        The app expects a CSV file with columns like:
        - ID: Customer identifier
        - Year_Birth: Customer birth year
        - Income: Customer income
        - Education: Education level
        - Marital_Status: Marital status
        - Kidhome/Teenhome: Number of kids/teens at home
        - Dt_Customer: Date of customer enrollment
        - MntWines, MntFruits, etc.: Spending in product categories
        - NumDealsPurchases, NumWebPurchases, etc.: Number of purchases by channel
        - AcceptedCmp1-5: Whether customer accepted previous campaigns (0/1)
        - Response: Target variable - response to campaign (0/1)
        """)
    
    # Create sidebar for data upload and processing
    with st.sidebar:
        st.header("Data Processing")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Marketing Campaign CSV", type=['csv'])
        
        if uploaded_file is not None:
            # Delimiter selection
            delimiter = st.selectbox(
                "Select CSV delimiter",
                options=[';', ',', '\t', '|'],
                index=0,
                help="Choose the delimiter used in your CSV file"
            )
            
            # Load data button
            load_button = st.button("Load Data")
            
            if load_button:
                with st.spinner("Loading and processing data..."):
                    # Load data
                    df = load_data(uploaded_file, delimiter)
                    
                    if df is not None:
                        # Store in session state
                        st.session_state.raw_data = df
                        st.session_state.step = 1
                        st.rerun()
        
        # Show processing options if data is loaded
        if 'raw_data' in st.session_state:
            st.success("Data loaded successfully")
            
            # Data processing steps
            if st.session_state.get('step', 0) >= 1:
                if st.button("Clean & Transform Data"):
                    with st.spinner("Cleaning and transforming data..."):
                        # Validate and clean data
                        df_clean = clean_data(st.session_state.raw_data)
                        st.session_state.clean_data = df_clean
                        
                        # Create features
                        df_features = create_features(df_clean)
                        st.session_state.featured_data = df_features
                        
                        st.session_state.step = 2
                        st.rerun()
            
            # Model training
            if st.session_state.get('step', 0) >= 2:
                if 'Response' in st.session_state.featured_data.columns:
                    if st.button("Train Prediction Model"):
                        with st.spinner("Training model... This may take a few minutes"):
                            model, results, important_features = train_model(
                                st.session_state.featured_data, 
                                target_col='Response'
                            )
                            
                            if model is not None:
                                st.session_state.model = model
                                st.session_state.model_results = results
                                st.session_state.important_features = important_features
                                st.session_state.step = 3
                                st.rerun()
                else:
                    st.info("'Response' column not found - model training disabled")

    # Main content area
    if 'raw_data' not in st.session_state:
        # Show sample dataset when no data is loaded
        st.info("👈 Please upload your marketing campaign data file using the sidebar")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Expected data format:
            Your CSV file should include customer information and campaign responses.
            Mandatory columns include:
            - Customer demographics (Year_Birth, Income, etc.)
            - Purchase behavior (spending by category)
            - Campaign responses
            """)
        
        with col2:
            st.markdown("""
            ### What you'll get:
            1. Customer segmentation analysis
            2. Spending patterns and profiles
            3. Campaign effectiveness insights
            4. Response prediction model
            5. Actionable recommendations
            """)
    
    else:
        # Create tabs for different analyses
        tabs = st.tabs([
            "Data Overview", 
            "Customer Segments", 
            "Spending Analysis", 
            "Campaign Analysis",
            "Response Prediction"
        ])
        
        # ------- Tab 1: Data Overview -------
        with tabs[0]:
            st.header("Data Overview")

            if st.session_state.get('step', 0) >= 2:
                display_df = st.session_state.featured_data
            else:
                display_df = st.session_state.raw_data

            # Display dataframe with pagination - ensure it's Arrow compatible
            arrow_safe_df = make_arrow_compatible(display_df.head(100))
            st.dataframe(arrow_safe_df)
            
            # Download link for the displayed dataframe
            if st.session_state.get('step', 0) >= 2:
                st.markdown(get_download_link(display_df), unsafe_allow_html=True)
            
            # Show data summary
            with st.expander("Data Summary Statistics"):
                st.write(display_df.select_dtypes(include=['number']).describe())
            
            # Data visualizations
            st.subheader("Data Visualizations")
            
            # Only show these visualizations if we have processed data
            if st.session_state.get('step', 0) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution
                    if 'Age' in display_df.columns:
                        fig = px.histogram(
                            display_df, 
                            x='Age',
                            title='Customer Age Distribution',
                            labels={'Age': 'Age'},
                            nbins=20,
                            opacity=0.7
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Income distribution
                    if 'Income' in display_df.columns:
                        fig = px.histogram(
                            display_df, 
                            x='Income',
                            title='Customer Income Distribution',
                            labels={'Income': 'Income'},
                            nbins=20,
                            opacity=0.7
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlations
                numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:
                    st.subheader("Feature Correlations")
                    
                    # Limit number of columns to avoid performance issues
                    if len(numeric_cols) > 15:
                        selected_numeric = [col for col in numeric_cols if 'Proportion' not in col][:15]
                    else:
                        selected_numeric = numeric_cols
                    
                    corr_matrix = display_df[selected_numeric].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # ------- Tab 2: Customer Segments -------
        with tabs[1]:
            st.header("Customer Segmentation")
            
            if st.session_state.get('step', 0) < 2:
                st.info("Please clean and transform your data to view customer segments")
            else:
                df = st.session_state.featured_data
                
                # Customer segments based on RFM
                if 'RFM_Segment' in df.columns:
                    st.subheader("RFM Segments")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        segment_counts = df['RFM_Segment'].value_counts().reset_index()
                        segment_counts.columns = ['Segment', 'Count']
                        
                        fig = px.pie(
                            segment_counts,
                            values='Count',
                            names='Segment',
                            title='Customer RFM Segments',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'Response' in df.columns:
                            response_by_segment = df.groupby('RFM_Segment')['Response'].mean().reset_index()
                            response_by_segment.columns = ['Segment', 'Response Rate']
                            response_by_segment['Response Rate'] = response_by_segment['Response Rate'] * 100
                            
                            response_by_segment = response_by_segment.sort_values('Response Rate', ascending=False)
                            
                            fig = px.bar(
                                response_by_segment,
                                x='Segment',
                                y='Response Rate',
                                title='Campaign Response Rate by RFM Segment',
                                labels={'Response Rate': 'Response Rate (%)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Customer segments based on Demographics
                st.subheader("Demographic Segments")
                
                demographic_cols = []
                
                if 'Generation' in df.columns:
                    demographic_cols.append('Generation')
                if 'Education' in df.columns:
                    demographic_cols.append('Education')
                if 'Marital_Status' in df.columns:
                    demographic_cols.append('Marital_Status')
                if 'LifeStage' in df.columns:
                    demographic_cols.append('LifeStage')
                
                if demographic_cols:
                    select_demographic = st.selectbox(
                        "Select demographic dimension:", 
                        options=demographic_cols
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        demo_counts = df[select_demographic].value_counts().reset_index()
                        demo_counts.columns = [select_demographic, 'Count']
                        
                        fig = px.bar(
                            demo_counts,
                            x=select_demographic,
                            y='Count',
                            title=f'Customer Counts by {select_demographic}',
                            labels={'Count': 'Number of Customers'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'TotalSpend' in df.columns:
                            spend_by_demo = df.groupby(select_demographic)['TotalSpend'].mean().reset_index()
                            spend_by_demo.columns = [select_demographic, 'Average Spend']
                            
                            fig = px.bar(
                                spend_by_demo,
                                x=select_demographic,
                                y='Average Spend',
                                title=f'Average Spend by {select_demographic}',
                                labels={'Average Spend': 'Average Spend ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Customer clustering 
                if 'TotalSpend' in df.columns and 'AvgSpend' in df.columns:
                    st.subheader("Customer Value vs. Loyalty")
                    
                    # Create a color column - use RFM_Segment if available, else use Generation
                    color_col = 'RFM_Segment' if 'RFM_Segment' in df.columns else ('Generation' if 'Generation' in df.columns else None)
                    
                    if color_col:
                        fig = px.scatter(
                            df,
                            x='TotalSpend',
                            y='AvgSpend',
                            color=color_col,
                            hover_name=color_col,
                            title='Customer Value vs. Average Purchase',
                            labels={
                                'TotalSpend': 'Total Customer Spend ($)',
                                'AvgSpend': 'Average Purchase Value ($)'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                        if 'Response' in df.columns:
                            st.markdown("""
                            💡 **Insight:** Identify high-value customer segments most likely to respond to campaigns.
                            These customers represent your best targeting opportunities.
                            """)
        
        # ------- Tab 3: Spending Analysis -------
        with tabs[2]:
            st.header("Customer Spending Analysis")
            
            if st.session_state.get('step', 0) < 2:
                st.info("Please clean and transform your data to view spending analysis")
            else:
                df = make_arrow_compatible(st.session_state.featured_data)
                
                # Product category spending
                st.subheader("Spending by Product Category")
                
                spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
                
                existing_spend_cols = [col for col in spending_cols if col in df.columns]
                
                if existing_spend_cols:
                    # Melt the spending data for visualization
                    spend_data = df[existing_spend_cols].sum().reset_index()
                    spend_data.columns = ['Category', 'Total Spend']
                    
                    # Clean category names for display
                    spend_data['Category'] = spend_data['Category'].str.replace('Mnt', '')
                    spend_data['Category'] = spend_data['Category'].str.replace('Products', '')
                    spend_data['Category'] = spend_data['Category'].str.replace('Prods', '')
                    
                    fig = px.pie(
                        spend_data, 
                        values='Total Spend', 
                        names='Category',
                        title='Total Spending by Product Category',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Spending by demographic segment
                    if 'Generation' in df.columns and 'TotalSpend' in df.columns:
                        st.subheader("Spending Analysis by Generation")
                        
                        pivot_data = pd.DataFrame()
                        
                        for col in existing_spend_cols:
                            category = col.replace('Mnt', '').replace('Products', '').replace('Prods', '')
                            temp_df = df.groupby('Generation')[col].mean().reset_index()
                            temp_df['Category'] = category
                            temp_df['Average Spend'] = temp_df[col]
                            pivot_data = pd.concat([pivot_data, temp_df])
                        
                        fig = px.bar(
                            pivot_data,
                            x='Generation',
                            y='Average Spend',
                            color='Category',
                            barmode='group',
                            title='Average Spending by Generation and Category',
                            labels={'Average Spend': 'Average Spend ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Purchase channels analysis
                purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
                existing_purchase_cols = [col for col in purchase_cols if col in df.columns]
                
                if existing_purchase_cols:
                    st.subheader("Purchase Channel Analysis")
                    
                    # Channel distribution
                    channel_data = df[existing_purchase_cols].sum().reset_index()
                    channel_data.columns = ['Channel', 'Number of Purchases']
                    
                    # Clean channel names
                    channel_data['Channel'] = channel_data['Channel'].str.replace('Num', '')
                    channel_data['Channel'] = channel_data['Channel'].str.replace('Purchases', '')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            channel_data,
                            values='Number of Purchases',
                            names='Channel',
                            title='Purchase Distribution by Channel',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'PreferredChannel' in df.columns:
                            channel_pref = df['PreferredChannel'].value_counts().reset_index()
                            channel_pref.columns = ['Preferred Channel', 'Count']
                            
                            fig = px.bar(
                                channel_pref,
                                x='Preferred Channel',
                                y='Count',
                                title='Customer Preferred Channel'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Channel usage by segment
                    segment_col = 'RFM_Segment' if 'RFM_Segment' in df.columns else ('Generation' if 'Generation' in df.columns else None)
                    
                    if segment_col:
                        st.subheader(f"Channel Usage by {segment_col}")
                        
                        pivot_data = pd.DataFrame()
                        
                        for col in existing_purchase_cols:
                            channel = col.replace('Num', '').replace('Purchases', '')
                            temp_df = df.groupby(segment_col)[col].mean().reset_index()
                            temp_df['Channel'] = channel
                            temp_df['Average Purchases'] = temp_df[col]
                            pivot_data = pd.concat([pivot_data, temp_df])
                        
                        fig = px.bar(
                            pivot_data,
                            x=segment_col,
                            y='Average Purchases',
                            color='Channel',
                            barmode='group',
                            title=f'Average Purchase Count by {segment_col} and Channel'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # ------- Tab 4: Campaign Analysis -------
        with tabs[3]:
            st.header("Campaign Performance Analysis")
            
            if st.session_state.get('step', 0) < 2:
                st.info("Please clean and transform your data to view campaign analysis")
            else:
                df = make_arrow_compatible(st.session_state.featured_data)
                
                # Campaign acceptance rates
                campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                'AcceptedCmp4', 'AcceptedCmp5']
                
                existing_campaign_cols = [col for col in campaign_cols if col in df.columns]
                
                if existing_campaign_cols:
                    campaign_rates = []
                    
                    for col in existing_campaign_cols:
                        campaign_num = col.replace('AcceptedCmp', '')
                        acceptance_rate = df[col].mean() * 100
                        campaign_rates.append({
                            'Campaign': f'Campaign {campaign_num}',
                            'Acceptance Rate': acceptance_rate
                        })
                    
                    campaign_df = pd.DataFrame(campaign_rates)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Campaign Acceptance Rates")
                        
                        fig = px.bar(
                            campaign_df,
                            x='Campaign',
                            y='Acceptance Rate',
                            title='Campaign Acceptance Rates',
                            labels={'Acceptance Rate': 'Acceptance Rate (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Total Campaign Responses")
                        
                        if 'TotalAcceptedCampaigns' in df.columns:
                            response_counts = df['TotalAcceptedCampaigns'].value_counts().sort_index().reset_index()
                            response_counts.columns = ['Number of Accepted Campaigns', 'Count']
                            
                            fig = px.pie(
                                response_counts,
                                values='Count',
                                names='Number of Accepted Campaigns',
                                title='Campaigns Accepted per Customer',
                                hole=0.4
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Campaign response by demographic
                    st.subheader("Campaign Response Analysis by Segment")
                    
                    demo_cols = []
                    if 'Generation' in df.columns:
                        demo_cols.append('Generation')
                    if 'Education' in df.columns:
                        demo_cols.append('Education')
                    if 'Marital_Status' in df.columns:
                        demo_cols.append('Marital_Status')
                    if 'RFM_Segment' in df.columns:
                        demo_cols.append('RFM_Segment')
                    
                    if demo_cols:
                        selected_demo = st.selectbox(
                            "Select segment dimension:",
                            options=demo_cols,
                            key="campaign_demo_select"
                        )
                        
                        if selected_demo:
                            # Campaign acceptance rates by segment
                            pivot_data = pd.DataFrame()
                            
                            for col in existing_campaign_cols:
                                campaign_num = col.replace('AcceptedCmp', '')
                                temp_df = df.groupby(selected_demo)[col].mean().reset_index()
                                temp_df['Campaign'] = f'Campaign {campaign_num}'
                                temp_df['Acceptance Rate'] = temp_df[col] * 100
                                pivot_data = pd.concat([pivot_data, temp_df])
                            
                            fig = px.bar(
                                pivot_data,
                                x=selected_demo,
                                y='Acceptance Rate',
                                color='Campaign',
                                barmode='group',
                                title=f'Campaign Acceptance Rates by {selected_demo}',
                                labels={'Acceptance Rate': 'Acceptance Rate (%)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Last campaign response analysis (if available)
                if 'Response' in df.columns:
                    st.subheader("Last Campaign Response Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Overall response rate
                        response_rate = df['Response'].mean() * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=response_rate,
                            title={'text': "Last Campaign Response Rate"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 20], 'color': "lightgray"},
                                    {'range': [20, 40], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': response_rate
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Response counts
                        response_counts = df['Response'].value_counts().reset_index()
                        response_counts.columns = ['Response', 'Count']
                        response_counts['Response'] = response_counts['Response'].map({0: 'No Response', 1: 'Responded'})
                        
                        fig = px.pie(
                            response_counts,
                            values='Count',
                            names='Response',
                            title='Response Distribution',
                            hole=0.4,
                            color_discrete_map={'No Response': 'lightgray', 'Responded': 'darkblue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Response rate by segment
                    if demo_cols:
                        selected_response_demo = st.selectbox(
                            "Select dimension for response analysis:",
                            options=demo_cols,
                            key="response_demo_select"
                        )
                        
                        response_by_demo = df.groupby(selected_response_demo)['Response'].mean().reset_index()
                        response_by_demo.columns = [selected_response_demo, 'Response Rate']
                        response_by_demo['Response Rate'] = response_by_demo['Response Rate'] * 100
                        response_by_demo = response_by_demo.sort_values('Response Rate', ascending=False)
                        
                        fig = px.bar(
                            response_by_demo,
                            x=selected_response_demo,
                            y='Response Rate',
                            title=f'Response Rate by {selected_response_demo}',
                            labels={'Response Rate': 'Response Rate (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # ------- Tab 5: Response Prediction -------
        with tabs[4]:
            st.header("Campaign Response Prediction")
            
            if st.session_state.get('step', 0) < 3:
                st.info("Please train the prediction model to view this section")
            elif 'model' not in st.session_state:
                st.warning("Model training was not successful. Please try again.")
            else:
                model = st.session_state.model
                results = st.session_state.model_results
                important_features = st.session_state.important_features
                
                # Model performance metrics
                st.subheader("Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.2%}")
                with col4:
                    st.metric("ROC AUC", f"{results['roc_auc']:.2%}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    cm = results['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted No', 'Predicted Yes'],
                        y=['Actual No', 'Actual Yes'],
                        hoverongaps=False,
                        colorscale='Blues'
                    ))
                    fig.update_layout(title='Confusion Matrix')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        feature_imp = pd.DataFrame({
                            'Feature': important_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig = px.bar(
                            feature_imp,
                            x='Importance',
                            y='Feature',
                            title='Top 10 Most Important Features',
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                y_test = results['y_test']
                y_pred_proba = results['y_pred_proba'] 
                
                # Create arrays for ROC curve
                y_test_array = np.array(y_test)
                y_pred_array = np.array(y_pred_proba)
                
                fpr, tpr, _ = roc_curve(y_test_array, y_pred_array)
                auc_score = auc(fpr, tpr)
                
                fig = px.line(
                    x=fpr, y=tpr,
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                    title=f'ROC Curve (AUC = {auc_score:.3f})'
                )
                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Customer targeting recommendations
                st.subheader("Campaign Targeting Recommendations")
                
                st.markdown("""
                Based on the predictive model, consider the following targeting strategies:
                
                1. **High-probability customers**: Target customers with the highest predicted response probability
                2. **Look-alike targeting**: Find customers similar to those who have responded positively before
                3. **Feature-based targeting**: Focus on customers with specific values for the most important features
                """)
                
                if 'RFM_Segment' in st.session_state.featured_data.columns:
                    # Get model predictions and probabilities
                    df = make_arrow_compatible(st.session_state.featured_data.copy())
                    
                    # Add predictions if missing
                    if 'Predicted_Response' not in df.columns:
                        try:
                            # Prepare features
                            X = df.drop(columns=['ID', 'Response'] if 'ID' in df.columns else ['Response'])
                            
                            # Handle datetime features that might be missing
                            datetime_features = results.get('datetime_features', {})
                            for dt_col, feature_cols in datetime_features.items():
                                if dt_col in df.columns and pd.api.types.is_datetime64_dtype(df[dt_col]):
                                    # Create the datetime derived features
                                    X[f'{dt_col}_Year'] = df[dt_col].dt.year
                                    X[f'{dt_col}_Month'] = df[dt_col].dt.month
                                    X[f'{dt_col}_Day'] = df[dt_col].dt.day
                            
                            # Handle categorical features
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            if cat_cols:
                                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                            
                            # Handle missing values
                            X = X.fillna(X.median())
                            
                            # Make sure all required features exist in X
                            missing_features = [col for col in important_features if col not in X.columns]
                            if missing_features:
                                st.warning(f"Some features used in the model are missing in the data. Adding zeros for: {missing_features}")
                                for col in missing_features:
                                    X[col] = 0
                            
                            # Only select the features used by the model
                            X_selected = X[important_features]
                            
                            # Predict
                            y_pred = model.predict(X_selected)
                            y_prob = model.predict_proba(X_selected)[:, 1]
                            
                            df['Predicted_Response'] = y_pred
                            df['Response_Probability'] = y_prob
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.info("Please retrain the model to fix this issue.")
                    
                    # Show response rate by segment
                    response_by_segment = df.groupby('RFM_Segment')['Response_Probability'].mean().reset_index()
                    response_by_segment.columns = ['RFM_Segment', 'Average Response Probability']
                    response_by_segment = response_by_segment.sort_values('Average Response Probability', ascending=False)
                    
                    fig = px.bar(
                        response_by_segment,
                        x='RFM_Segment',
                        y='Average Response Probability',
                        title='Predicted Response Probability by RFM Segment',
                        labels={'Average Response Probability': 'Avg. Response Probability'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    💡 **Key Insight**: Based on the model predictions, focusing your marketing efforts 
                    on the top segments could significantly improve campaign ROI.
                    """)

if __name__ == "__main__":
    main()