import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Optimize for low memory
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Page config
st.set_page_config(
    page_title="Empty Forecasting",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Empty Shipments Forecasting")
st.markdown("Upload your data to forecast empty shipments")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    forecast_days = st.slider("Forecast Days Ahead", 1, 30, 7)
    st.markdown("---")
    st.markdown("### 📥 Download Template")
    
    # Generate template
    product_template = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'customer_id': ['CUST_001', 'CUST_001'],
        'product_type': ['soft_drink', 'water'],
        'quantity': [100, 50]
    })
    
    empty_template = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'customer_id': ['CUST_001', 'CUST_001'],
        'empty_type': ['pallets', 'pallets'],
        'opening_balance': [50, 53],
        'received_qty': [5, 4],
        'issued_qty': [2, 3],
        'closing_balance': [53, 54]
    })
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        product_template.to_excel(writer, sheet_name='Products', index=False)
        empty_template.to_excel(writer, sheet_name='Empties', index=False)
    
    st.download_button(
        "📄 Download Template",
        data=output.getvalue(),
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# File uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Product Data")
    product_file = st.file_uploader("Upload Products Excel", type=['xlsx', 'xls'], key='prod')

with col2:
    st.subheader("📦 Empty Data")
    empty_file = st.file_uploader("Upload Empties Excel", type=['xlsx', 'xls'], key='emp')

if product_file and empty_file:
    try:
        # Load data
        product_df = pd.read_excel(product_file)
        empty_df = pd.read_excel(empty_file)
        
        # Validate
        req_prod_cols = ['date', 'customer_id', 'product_type', 'quantity']
        req_empty_cols = ['date', 'customer_id', 'empty_type', 'opening_balance', 
                         'received_qty', 'issued_qty', 'closing_balance']
        
        if not all(col in product_df.columns for col in req_prod_cols):
            st.error("❌ Product file missing required columns")
            st.stop()
        
        if not all(col in empty_df.columns for col in req_empty_cols):
            st.error("❌ Empty file missing required columns")
            st.stop()
        
        st.success("✅ Files loaded successfully!")
        
        # Show previews
        with st.expander("👀 Preview Data"):
            tab1, tab2 = st.tabs(["Products", "Empties"])
            with tab1:
                st.dataframe(product_df.head())
            with tab2:
                st.dataframe(empty_df.head())
        
        # Select empty type
        empty_types = empty_df['empty_type'].unique()
        selected_type = st.selectbox("Select Empty Type", empty_types)
        
        # Train button
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Training model..."):
                
                # Convert dates
                product_df['date'] = pd.to_datetime(product_df['date'])
                empty_df['date'] = pd.to_datetime(empty_df['date'])
                
                # Filter empty type
                empty_filtered = empty_df[empty_df['empty_type'] == selected_type].copy()
                
                # Aggregate products
                product_agg = product_df.groupby(['date', 'customer_id', 'product_type'])['quantity'].sum().reset_index()
                product_pivot = product_agg.pivot_table(
                    index=['date', 'customer_id'],
                    columns='product_type',
                    values='quantity',
                    fill_value=0
                ).reset_index()
                
                # Merge
                merged = empty_filtered.merge(product_pivot, on=['date', 'customer_id'], how='left').fillna(0)
                merged = merged.sort_values(['customer_id', 'date'])
                
                # Simple feature engineering
                features = merged.copy()
                features['day_of_week'] = features['date'].dt.dayofweek
                features['month'] = features['date'].dt.month
                
                # Add lag features (lightweight)
                customer_groups = features.groupby('customer_id')
                
                for lag in [7, 14]:
                    if 'soft_drink' in features.columns:
                        features[f'soft_drink_lag_{lag}'] = customer_groups['soft_drink'].shift(lag)
                    if 'water' in features.columns:
                        features[f'water_lag_{lag}'] = customer_groups['water'].shift(lag)
                    features[f'issued_lag_{lag}'] = customer_groups['issued_qty'].shift(lag)
                
                # Rolling mean (7 days only)
                if 'soft_drink' in features.columns:
                    features['soft_drink_ma7'] = customer_groups['soft_drink'].transform(
                        lambda x: x.rolling(7, min_periods=1).mean()
                    )
                if 'water' in features.columns:
                    features['water_ma7'] = customer_groups['water'].transform(
                        lambda x: x.rolling(7, min_periods=1).mean()
                    )
                features['issued_ma7'] = customer_groups['issued_qty'].transform(
                    lambda x: x.rolling(7, min_periods=1).mean()
                )
                
                # Target
                features['target'] = customer_groups['issued_qty'].shift(-forecast_days)
                features = features.dropna(subset=['target'])
                
                # Select features
                feature_cols = [col for col in features.columns if col not in 
                               ['date', 'customer_id', 'empty_type', 'target', 
                                'opening_balance', 'received_qty', 'issued_qty', 'closing_balance']]
                
                X = features[feature_cols].fillna(0)
                y = features['target']
                
                # Train-test split (80-20)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Lightweight model
                model = RandomForestRegressor(
                    n_estimators=50,  # Reduced for speed
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=1  # Single thread for memory efficiency
                )
                
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Metrics
                st.subheader("📈 Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test MAE", f"{mean_absolute_error(y_test, test_pred):.2f}")
                
                with col2:
                    st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")
                
                with col3:
                    accuracy = 1 - mean_absolute_error(y_test, test_pred) / (y_test.mean() + 1)
                    st.metric("Accuracy", f"{accuracy*100:.1f}%")
                
                # Generate all predictions
                predictions = model.predict(X)
                features['predicted'] = predictions
                
                # Summary by customer
                st.subheader("📊 Forecast Summary")
                
                summary = features.groupby('customer_id').agg({
                    'target': 'sum',
                    'predicted': 'sum',
                    'closing_balance': 'last'
                }).reset_index()
                summary.columns = ['Customer', 'Actual', 'Forecast', 'Current Stock']
                summary['Forecast'] = summary['Forecast'].round(0).astype(int)
                
                st.dataframe(summary, use_container_width=True)
                
                # Customer detail
                st.subheader("🔍 Customer Details")
                selected_customer = st.selectbox("Select Customer", features['customer_id'].unique())
                
                customer_data = features[features['customer_id'] == selected_customer].tail(30)
                
                # Simple line chart
                chart_data = customer_data[['date', 'target', 'predicted']].set_index('date')
                chart_data.columns = ['Actual', 'Forecast']
                st.line_chart(chart_data)
                
                # Download results
                st.subheader("💾 Download Results")
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    features[['date', 'customer_id', 'closing_balance', 'target', 'predicted']].to_excel(
                        writer, sheet_name='Forecasts', index=False
                    )
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    "📥 Download Forecast Excel",
                    data=output.getvalue(),
                    file_name=f"forecast_{selected_type}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload both files to get started")

# Footer
st.markdown("---")
st.markdown("*Optimized for Render Free Tier*")
