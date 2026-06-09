import os
import pandas as pd

def load_sales(data_dir='data/raw/'):
    """Loads sales data and parses Date."""
    path = os.path.join(data_dir, 'sales.csv')
    return pd.read_csv(path, parse_dates=['Date'])

def load_orders(data_dir='data/raw/'):
    """Loads orders data and parses order_date."""
    path = os.path.join(data_dir, 'orders.csv')
    return pd.read_csv(path, parse_dates=['order_date'])

def load_order_items(data_dir='data/raw/'):
    """Loads order items data."""
    path = os.path.join(data_dir, 'order_items.csv')
    return pd.read_csv(path)

def load_customers(data_dir='data/raw/'):
    """Loads customer database and parses signup_date."""
    path = os.path.join(data_dir, 'customers.csv')
    return pd.read_csv(path, parse_dates=['signup_date'])

def load_geography(data_dir='data/raw/'):
    """Loads geography details."""
    path = os.path.join(data_dir, 'geography.csv')
    return pd.read_csv(path)

def load_promotions(data_dir='data/raw/'):
    """Loads promotions schedule and parses start_date and end_date."""
    path = os.path.join(data_dir, 'promotions.csv')
    return pd.read_csv(path, parse_dates=['start_date', 'end_date'])

def load_returns(data_dir='data/raw/'):
    """Loads returns data."""
    path = os.path.join(data_dir, 'returns.csv')
    return pd.read_csv(path)

def load_all_datasets(data_dir='data/raw/'):
    """Loads all data science datasets and returns them as a dictionary."""
    return {
        'sales': load_sales(data_dir),
        'orders': load_orders(data_dir),
        'order_items': load_order_items(data_dir),
        'customers': load_customers(data_dir),
        'geography': load_geography(data_dir),
        'promotions': load_promotions(data_dir),
        'returns': load_returns(data_dir)
    }
