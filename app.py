import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Empty Shipments Forecasting",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class EmptyShipmentsForecaster:
    """ML model for forecasting empty shipments."""
    
    def __init__(self, empty_type='pallets'):
        self.empty_type = empty_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_data(self, product_shipments_df, empty_transactions_df, 
                     forecast_horizon=7):
        """Prepare and engineer features from raw data."""
        # Ensure date columns are datetime
        product_shipments_df['date'] = pd.to_datetime(product_shipments_df['date'])
        empty_transactions_df['date'] = pd.to_datetime(empty_transactions_df['date'])
        
        # Filter for specific empty type
        empty_data = empty_transactions_df[
            empty_transactions_df['empty_type'] == self.empty_type
        ].copy()
        
        # Aggregate product shipments by customer and date
        product_agg = product_shipments_df.groupby(
            ['date', 'customer_id', 'product_type']
        )['quantity'].sum().reset_index()
        
        # Pivot to get soft_drink and water as separate columns
        product_pivot = product_agg.pivot_table(
            index=['date', 'customer_id'],
            columns='product_type',
            values='quantity',
            fill_value=0
        ).reset_index()
        
        # Merge product shipments with empty transactions
        merged = empty_data.merge(
            product_pivot,
            on=['date', 'customer_id'],
            how='left'
        )
        
        # Fill NaN values
        merged = merged.fillna(0)
        
        # Sort by customer and date
        merged = merged.sort_values(['customer_id', 'date'])
        
        # Feature engineering
        features_df = self._engineer_features(merged, forecast_horizon)
        
        return features_df
    
    def _engineer_features(self, df, forecast_horizon):
        """Create lag features, rolling statistics, and temporal features."""
        features = df.copy()
        
        # Temporal features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['day_of_month'] = features['date'].dt.day
        features['month'] = features['date'].dt.month
        features['quarter'] = features['date'].dt.quarter
        features['week_of_year'] = features['date'].dt.isocalendar().week
        
        # Calculate empty generation rate (issued qty)
        features['empty_generation'] = features['issued_qty']
        
        # Group by customer for time-series features
        customer_groups = features.groupby('customer_id')
        
        # Lag features for product shipments
        for col in ['soft_drink', 'water']:
            if col in features.columns:
                for lag in [1, 7, 14, 21, 28]:
                    features[f'{col}_lag_{lag}'] = customer_groups[col].shift(lag)
        
        # Lag features for empty transactions
        for lag in [1, 7, 14, 21]:
            features[f'issued_qty_lag_{lag}'] = customer_groups['issued_qty'].shift(lag)
            features[f'received_qty_lag_{lag}'] = customer_groups['received_qty'].shift(lag)
            features[f'closing_balance_lag_{lag}'] = customer_groups['closing_balance'].shift(lag)
        
        # Rolling statistics (7, 14, 28 days)
        for window in [7, 14, 28]:
            for col in ['soft_drink', 'water']:
                if col in features.columns:
                    features[f'{col}_rolling_mean_{window}'] = customer_groups[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    features[f'{col}_rolling_std_{window}'] = customer_groups[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
            
            features[f'issued_qty_rolling_mean_{window}'] = customer_groups['issued_qty'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'issued_qty_rolling_std_{window}'] = customer_groups['issued_qty'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Ratio features
        if 'soft_drink' in features.columns and 'water' in features.columns:
            features['total_product'] = features['soft_drink'] + features['water']
            features['water_ratio'] = features['water'] / (features['total_product'] + 1)
        
        # Empty stock utilization
        features['stock_turnover'] = features['issued_qty'] / (features['opening_balance'] + 1)
        
        # Customer-level aggregations
        customer_stats = features.groupby('customer_id').agg({
            'issued_qty': ['mean', 'std'],
            'closing_balance': ['mean', 'std'],
            'soft_drink': 'mean',
            'water': 'mean'
        }).reset_index()
        customer_stats.columns = ['customer_id', 'customer_issued_mean', 'customer_issued_std',
                                   'customer_balance_mean', 'customer_balance_std',
                                   'customer_soft_drink_mean', 'customer_water_mean']
        
        features = features.merge(customer_stats, on='customer_id', how='left')
        
        # Target variable
        features['target'] = customer_groups['issued_qty'].shift(-forecast_horizon)
        
        # Drop rows with NaN in target
        features = features.dropna(subset=['target'])
        
        return features
    
    def train(self, features_df, model_type='random_forest', test_size=0.2):
        """Train the forecasting model."""
        exclude_cols = ['date', 'customer_id', 'empty_type', 'target', 
                       'opening_balance', 'received_qty', 'issued_qty', 'closing_balance']
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Train-test split
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                min_samples_split=10,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
        }
        
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importance'] = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return metrics
    
    def predict(self, features_df):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions


def validate_data_format(df, required_columns, file_type):
    """Validate uploaded data format."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        st.error(f"❌ {file_type} file is missing columns: {missing_cols}")
        return False
    return True


def download_excel_template():
    """Generate Excel template files for download."""
    # Product shipments template
    product_template = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'customer_id': ['CUST_001', 'CUST_001', 'CUST_002'],
        'product_type': ['soft_drink', 'water', 'soft_drink'],
        'quantity': [100, 50, 80]
    })
    
    # Empty transactions template
    empty_template = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'customer_id': ['CUST_001', 'CUST_001', 'CUST_002'],
        'empty_type': ['pallets', 'racks', 'pallets'],
        'opening_balance': [50, 30, 45],
        'received_qty': [5, 3, 4],
        'issued_qty': [2, 1, 3],
        'closing_balance': [53, 32, 46]
    })
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        product_template.to_excel(writer, sheet_name='Product_Shipments', index=False)
        empty_template.to_excel(writer, sheet_name='Empty_Transactions', index=False)
    
    return output.getvalue()


def main():
    # Header
    st.markdown('<h1 class="main-header">📦 Empty Shipments Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Upload your historical data to forecast empty shipments (pallets, racks, shills) 
    based on product shipments and empty transaction history.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Download template
    st.sidebar.markdown("### 📥 Download Template")
    template_file = download_excel_template()
    st.sidebar.download_button(
        label="Download Excel Template",
        data=template_file,
        file_name="empty_forecast_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.sidebar.markdown("---")
    
    # Model parameters
    st.sidebar.markdown("### 🎯 Model Parameters")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
    model_type = st.sidebar.selectbox("Model Type", 
                                       ["random_forest", "gradient_boosting"])
    test_size = st.sidebar.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
    
    # File upload section
    st.header("📁 Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Shipments Data")
        st.markdown("*Required columns: date, customer_id, product_type, quantity*")
        product_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            key='product'
        )
    
    with col2:
        st.subheader("Empty Transactions Data")
        st.markdown("*Required columns: date, customer_id, empty_type, opening_balance, received_qty, issued_qty, closing_balance*")
        empty_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            key='empty'
        )
    
    # Process files
    if product_file is not None and empty_file is not None:
        try:
            # Read files
            product_df = pd.read_excel(product_file)
            empty_df = pd.read_excel(empty_file)
            
            # Validate formats
            product_cols = ['date', 'customer_id', 'product_type', 'quantity']
            empty_cols = ['date', 'customer_id', 'empty_type', 'opening_balance', 
                         'received_qty', 'issued_qty', 'closing_balance']
            
            if not validate_data_format(product_df, product_cols, "Product Shipments"):
                return
            if not validate_data_format(empty_df, empty_cols, "Empty Transactions"):
                return
            
            st.success("✅ Files uploaded successfully!")
            
            # Data preview
            with st.expander("📊 Preview Data"):
                tab1, tab2 = st.tabs(["Product Shipments", "Empty Transactions"])
                with tab1:
                    st.dataframe(product_df.head(10))
                    st.write(f"Total records: {len(product_df)}")
                with tab2:
                    st.dataframe(empty_df.head(10))
                    st.write(f"Total records: {len(empty_df)}")
            
            # Train model button
            st.markdown("---")
            st.header("🚀 Train & Forecast")
            
            empty_types = empty_df['empty_type'].unique()
            selected_empty_type = st.selectbox("Select Empty Type to Forecast", empty_types)
            
            if st.button("🎯 Train Model & Generate Forecasts", type="primary"):
                with st.spinner(f"Training model for {selected_empty_type}..."):
                    # Initialize forecaster
                    forecaster = EmptyShipmentsForecaster(empty_type=selected_empty_type)
                    
                    # Prepare data
                    features_df = forecaster.prepare_data(
                        product_df,
                        empty_df,
                        forecast_horizon=forecast_horizon
                    )
                    
                    # Train model
                    metrics = forecaster.train(features_df, 
                                              model_type=model_type,
                                              test_size=test_size)
                    
                    # Display metrics
                    st.success("✅ Model trained successfully!")
                    
                    st.subheader("📈 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Test MAE", f"{metrics['test_mae']:.2f}")
                        st.metric("Train MAE", f"{metrics['train_mae']:.2f}")
                    
                    with col2:
                        st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                        st.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
                    
                    with col3:
                        st.metric("Test R²", f"{metrics['test_r2']:.3f}")
                        st.metric("Train R²", f"{metrics['train_r2']:.3f}")
                    
                    # Feature importance
                    if 'feature_importance' in metrics and metrics['feature_importance'] is not None:
                        st.subheader("🎯 Top 15 Important Features")
                        importance_df = metrics['feature_importance'].head(15)
                        
                        fig = px.bar(importance_df, 
                                    x='importance', 
                                    y='feature',
                                    orientation='h',
                                    title='Feature Importance')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate forecasts
                    st.markdown("---")
                    st.subheader("📊 Forecasts by Customer")
                    
                    # Get predictions
                    predictions = forecaster.predict(features_df)
                    features_df['predicted_empty_shipment'] = predictions
                    
                    # Select customer for detailed view
                    customers = features_df['customer_id'].unique()
                    selected_customer = st.selectbox("Select Customer", customers)
                    
                    customer_data = features_df[
                        features_df['customer_id'] == selected_customer
                    ].copy()
                    
                    # Plot predictions vs actuals
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=customer_data['date'],
                        y=customer_data['target'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=customer_data['date'],
                        y=customer_data['predicted_empty_shipment'],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'Empty Shipments Forecast - {selected_customer}',
                        xaxis_title='Date',
                        yaxis_title='Empty Quantity',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast table
                    st.subheader("📋 Detailed Forecast")
                    forecast_display = customer_data[
                        ['date', 'closing_balance', 'target', 'predicted_empty_shipment']
                    ].tail(30).copy()
                    forecast_display.columns = ['Date', 'Current Stock', 
                                                'Actual Shipment', 'Predicted Shipment']
                    st.dataframe(forecast_display)
                    
                    # Summary by all customers
                    st.markdown("---")
                    st.subheader("📊 Summary - All Customers")
                    
                    summary = features_df.groupby('customer_id').agg({
                        'target': 'sum',
                        'predicted_empty_shipment': 'sum',
                        'closing_balance': 'last'
                    }).reset_index()
                    summary.columns = ['Customer', 'Actual Total', 
                                      'Predicted Total', 'Current Stock']
                    summary['Accuracy %'] = (1 - abs(summary['Actual Total'] - 
                                            summary['Predicted Total']) / 
                                            (summary['Actual Total'] + 1)) * 100
                    
                    st.dataframe(summary.style.format({
                        'Actual Total': '{:.0f}',
                        'Predicted Total': '{:.0f}',
                        'Current Stock': '{:.0f}',
                        'Accuracy %': '{:.1f}%'
                    }))
                    
                    # Download forecasts
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        features_df[['date', 'customer_id', 'closing_balance', 
                                    'target', 'predicted_empty_shipment']].to_excel(
                            writer, sheet_name='Forecasts', index=False)
                        summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download Forecast Results",
                        data=output.getvalue(),
                        file_name=f"forecast_{selected_empty_type}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload both Product Shipments and Empty Transactions files to continue.")


if __name__ == "__main__":
    main()
