import numpy as np
import pandas as pd

def compute_behavioral_averages(orders, order_items, traffic, sales):
    """Computes behavioral metrics mapped to calendar months (post-2019)."""
    # Parse dates
    traffic = traffic.copy()
    traffic['year'] = traffic['date'].dt.year
    traffic['month'] = traffic['date'].dt.month
    
    orders = orders.copy()
    orders['year'] = orders['order_date'].dt.year
    orders['month'] = orders['order_date'].dt.month

    # 1. Monthly Conversion & Sessions
    m_stats = pd.merge(
        traffic.groupby(['year', 'month'])['sessions'].sum().reset_index(),
        orders.groupby(['year', 'month'])['order_id'].count().reset_index(),
        on=['year', 'month']
    )
    m_stats['conv'] = m_stats['order_id'] / m_stats['sessions']
    avg_conv = m_stats[m_stats['year'] >= 2019].groupby('month')['conv'].mean().to_dict()
    avg_sessions = traffic[traffic['year'] >= 2019].groupby('month')['sessions'].mean().to_dict()

    # 2. Monthly Average Order Value (AOV)
    order_values = order_items.groupby('order_id')['unit_price'].sum().reset_index()
    orders_with_val = pd.merge(orders, order_values, on='order_id')
    avg_aov = orders_with_val[orders_with_val['year'] >= 2019].groupby('month')['unit_price'].mean().to_dict()

    # 3. Monthly Margin Rate
    sales = sales.copy()
    sales['margin_rate'] = (sales['Revenue'] - sales['COGS']) / sales['Revenue']
    avg_margin = sales[sales['Date'].dt.year >= 2019].groupby(sales['Date'].dt.month)['margin_rate'].mean().to_dict()

    return avg_conv, avg_sessions, avg_aov, avg_margin

def calculate_growth_and_baseline(sales):
    """Calculates YoY growth rates and daily baseline level in 2022."""
    sales = sales.copy()
    sales['year'] = sales['Date'].dt.year
    annual = sales.groupby('year')[['Revenue', 'COGS']].sum()
    recent_years = annual.loc[2020:2022]

    # Geometric mean growth rate
    growth_rev = (1 + recent_years['Revenue'].pct_change().dropna()).prod() ** (1/2)
    growth_cogs = (1 + recent_years['COGS'].pct_change().dropna()).prod() ** (1/2)

    # 2022 daily base
    base_rev_daily = annual.loc[2022, 'Revenue'] / 365
    base_cogs_daily = annual.loc[2022, 'COGS'] / 365

    return growth_rev, growth_cogs, base_rev_daily, base_cogs_daily

def apply_trend_normalization(df, growth, base_daily, target_col=None):
    """Calculates the trend baseline and normalizes the target variable (if provided)."""
    df = df.copy()
    df['years_ahead'] = df['Date'].dt.year - 2022
    df['trend'] = base_daily * (growth ** df['years_ahead'])
    
    if target_col and target_col in df.columns:
        df[f'{target_col}_norm'] = df[target_col] / df['trend']
    
    return df

def create_features(df, promos_df, avg_conv, avg_aov, avg_margin, avg_sessions):
    """Engineers all calendar, cyclical, behavioral, and promotion features."""
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dow'] = df['Date'].dt.dayofweek
    df['doy'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclical Time Features
    df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    
    # Behavioral Mappings
    df['feat_conv'] = df['month'].map(avg_conv)
    df['feat_aov'] = df['month'].map(avg_aov)
    df['feat_margin'] = df['month'].map(avg_margin)
    df['feat_sessions'] = df['month'].map(avg_sessions)
    
    # Domain-Specific Indicators
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['post_2019'] = (df['Date'].dt.year >= 2019).astype(int)
    df['is_payday_period'] = ((df['day'] >= 14) & (df['day'] <= 16) | (df['day'] >= 28)).astype(int)
    df['is_beginning_month'] = (df['day'] <= 5).astype(int)
    df['month_progress'] = df['day'] / df['Date'].dt.days_in_month
    
    # Promotion Context
    p_counts = []
    for d in df['Date']:
        p_counts.append(len(promos_df[(promos_df['start_date'] <= d) & (promos_df['end_date'] >= d)]))
    df['promo_count'] = p_counts
    
    return df
