import pandas as pd
import numpy as np
from datetime import datetime
import snowflake.snowpark as snowpark

def model(dbt, session: snowpark.Session):
    """
    Forecast next month's MAR (Monthly Active Rows) based on historical performance
    from the fivetran_platform__mar_table_history table.
    """
    
    # Configure the model
    dbt.config(
        materialized="table",
        snowflake_warehouse="DEFAULT"
    )
    
    # Read the historical MAR data
    mar_history = dbt.ref("fivetran_platform__mar_table_history")
    df = mar_history.to_pandas()
    
    # 🔑 Normalize column names to lowercase so we can use consistent names
    df.columns = df.columns.str.lower()
    
    # Filter for specific connector: connection_name = 'fivetran_log' and destination_id = 'durable_biased'
    filtered_df = df[
        (df['connection_name'] == 'fivetran_log') & 
        (df['destination_id'] == 'durable_biased')
    ]
    
    # Check if we have data for this connector
    if len(filtered_df) == 0:
        # No data found for this connector
        forecast_df = pd.DataFrame({
            'connection_name': ['fivetran_log'],
            'destination_id': ['durable_biased'],
            'forecast_month': [None],
            'forecasted_total_mar': [0],
            'forecast_lower_bound': [0],
            'forecast_upper_bound': [0],
            'forecast_method': ['no_data'],
            'historical_months_used': [0],
            'last_historical_month': [None],
            'forecast_created_at': [datetime.now()]
        })
        return forecast_df
    
    # Aggregate total MAR by month for this specific connector
    monthly_mar = (
        filtered_df.groupby('measured_month')['total_monthly_active_rows']
        .sum()
        .reset_index()
        .sort_values('measured_month')
    )
    
    # If somehow there's still no data after grouping, handle gracefully
    if monthly_mar.empty:
        forecast_df = pd.DataFrame({
            'connection_name': ['fivetran_log'],
            'destination_id': ['durable_biased'],
            'forecast_month': [None],
            'forecasted_total_mar': [0],
            'forecast_lower_bound': [0],
            'forecast_upper_bound': [0],
            'forecast_method': ['no_data'],
            'historical_months_used': [0],
            'last_historical_month': [None],
            'forecast_created_at': [datetime.now()]
        })
        return forecast_df
    
    # Convert measured_month to datetime if it's not already
    monthly_mar['measured_month'] = pd.to_datetime(monthly_mar['measured_month'])
    
    # Calculate next month's date
    last_month = monthly_mar['measured_month'].max()
    next_month = (last_month + pd.DateOffset(months=1)).replace(day=1)
    
    # Forecast logic
    if len(monthly_mar) >= 3:
        recent_months = monthly_mar.tail(6)  # Use last 6 months for trend
        
        x = np.arange(len(recent_months))
        y = recent_months['total_monthly_active_rows'].values
        
        coeffs = np.polyfit(x, y, 1)
        trend_slope = coeffs[0]
        intercept = coeffs[1]
        
        next_x = len(recent_months)
        forecast_mar = trend_slope * next_x + intercept
        
        ma_forecast = recent_months['total_monthly_active_rows'].tail(3).mean()
        
        final_forecast = 0.7 * forecast_mar + 0.3 * ma_forecast
        
        std_dev = recent_months['total_monthly_active_rows'].std()
        lower_bound = max(0, final_forecast - 1.96 * std_dev)
        upper_bound = final_forecast + 1.96 * std_dev
        
    elif len(monthly_mar) >= 1:
        final_forecast = monthly_mar['total_monthly_active_rows'].mean()
        std_dev = (
            monthly_mar['total_monthly_active_rows'].std()
            if len(monthly_mar) > 1
            else final_forecast * 0.1
        )
        lower_bound = max(0, final_forecast - 1.96 * std_dev)
        upper_bound = final_forecast + 1.96 * std_dev
    else:
        final_forecast = 0
        lower_bound = 0
        upper_bound = 0
        last_month = None
        next_month = None
    
    # Create forecast dataframe with connection details
    forecast_df = pd.DataFrame({
        'connection_name': ['fivetran_log'],
        'destination_id': ['durable_biased'],
        'forecast_month': [next_month],
        'forecasted_total_mar': [int(final_forecast)],
        'forecast_lower_bound': [int(lower_bound)],
        'forecast_upper_bound': [int(upper_bound)],
        'forecast_method': ['linear_trend_with_ma'],
        'historical_months_used': [len(monthly_mar)],
        'last_historical_month': [last_month],
        'forecast_created_at': [datetime.now()]
    })
    
    return forecast_df