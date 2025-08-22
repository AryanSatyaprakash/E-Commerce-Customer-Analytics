"""
Data Preprocessing Module for E-Commerce Customer Analytics

This module contains functions for cleaning and preprocessing the UCI Online Retail dataset.
It handles missing values, outliers, feature engineering, and data validation.

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import warnings
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceDataPreprocessor:
    """
    A comprehensive data preprocessing class for e-commerce transaction data.
    
    This class handles data loading, cleaning, validation, and feature engineering
    for retail transaction datasets.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_data(self, url: str = None) -> pd.DataFrame:
        """
        Load data from file or URL.
        
        Args:
            url (str): URL to download data from (UCI repository)
            
        Returns:
            pd.DataFrame: Raw dataframe
        """
        try:
            if url:
                logger.info(f"Downloading data from {url}")
                self.df = pd.read_excel(url)
            elif self.data_path:
                logger.info(f"Loading data from {self.data_path}")
                if self.data_path.endswith('.xlsx'):
                    self.df = pd.read_excel(self.data_path)
                elif self.data_path.endswith('.csv'):
                    self.df = pd.read_csv(self.data_path)
                else:
                    raise ValueError("Unsupported file format")
            else:
                # Create sample data for demonstration
                logger.info("Creating sample dataset for demonstration")
                self.df = self._create_sample_data()
                
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample e-commerce data for demonstration purposes.
        
        Returns:
            pd.DataFrame: Sample dataset with realistic e-commerce structure
        """
        np.random.seed(42)
        n_transactions = 10000
        
        # Generate realistic data
        invoice_nos = [f"INV{500000 + i}" for i in range(n_transactions)]
        stock_codes = [f"SKU{np.random.randint(1000, 2000)}" for _ in range(n_transactions)]
        
        products = [
            "JUMBO BAG RED RETROSPOT", "ALARM CLOCK BAKELIKE GREEN", "PARTY BUNTING",
            "LUNCH BAG RED RETROSPOT", "ASSORTED COLOUR BIRD ORNAMENT", "DOORMAT NEW ENGLAND",
            "SET OF 3 COLOURED FLYING DUCKS", "REGENT CAKESTAND 3 TIER", "SPACEBOY LUNCH BOX",
            "CREAM HANGING HEART T-LIGHT HOLDER", "RED TOADSTOOL LED NIGHT LIGHT",
            "KNITTED UNION FLAG HOT WATER BOTTLE", "RABBIT NIGHT LIGHT", "LONDON BUS PENCIL SHARPENER"
        ]
        
        countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium"]
        
        # Generate data
        data = {
            'InvoiceNo': invoice_nos,
            'StockCode': stock_codes,
            'Description': [np.random.choice(products) for _ in range(n_transactions)],
            'Quantity': np.random.randint(1, 50, n_transactions),
            'InvoiceDate': pd.date_range(start='2010-12-01', end='2011-12-09', periods=n_transactions),
            'UnitPrice': np.round(np.random.uniform(0.5, 25.0, n_transactions), 2),
            'CustomerID': np.random.randint(12000, 18000, n_transactions),
            'Country': [np.random.choice(countries, p=[0.85, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02]) 
                       for _ in range(n_transactions)]
        }
        
        df = pd.DataFrame(data)
        
        # Introduce some missing values and cancellations
        df.loc[np.random.choice(df.index, 1000), 'CustomerID'] = np.nan
        cancel_indices = np.random.choice(df.index, 200)
        df.loc[cancel_indices, 'InvoiceNo'] = df.loc[cancel_indices, 'InvoiceNo'].str.replace('INV', 'C')
        df.loc[cancel_indices, 'Quantity'] = -df.loc[cancel_indices, 'Quantity']
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning process...")
        
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.processed_df = self.df.copy()
        
        # Step 1: Basic data info
        self._log_data_info()
        
        # Step 2: Handle missing values
        self._handle_missing_values()
        
        # Step 3: Data type optimization
        self._optimize_data_types()
        
        # Step 4: Remove duplicates
        self._remove_duplicates()
        
        # Step 5: Handle outliers
        self._handle_outliers()
        
        # Step 6: Validate data integrity
        self._validate_data()
        
        logger.info("Data cleaning completed successfully!")
        return self.processed_df
    
    def _log_data_info(self):
        """Log basic information about the dataset."""
        logger.info(f"Dataset shape: {self.processed_df.shape}")
        logger.info(f"Columns: {list(self.processed_df.columns)}")
        logger.info(f"Missing values:\n{self.processed_df.isnull().sum()}")
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Remove rows with missing Description
        before_desc = len(self.processed_df)
        self.processed_df = self.processed_df.dropna(subset=['Description'])
        logger.info(f"Removed {before_desc - len(self.processed_df)} rows with missing Description")
        
        # Handle missing CustomerID (keep for guest transactions)
        missing_customers = self.processed_df['CustomerID'].isnull().sum()
        logger.info(f"Found {missing_customers} transactions without CustomerID (guest purchases)")
        
        # Remove rows with missing critical fields
        critical_fields = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice']
        before_critical = len(self.processed_df)
        self.processed_df = self.processed_df.dropna(subset=critical_fields)
        logger.info(f"Removed {before_critical - len(self.processed_df)} rows with missing critical fields")
    
    def _optimize_data_types(self):
        """Optimize data types for memory efficiency."""
        logger.info("Optimizing data types...")
        
        # Convert to appropriate data types
        self.processed_df['InvoiceDate'] = pd.to_datetime(self.processed_df['InvoiceDate'])
        self.processed_df['CustomerID'] = pd.to_numeric(self.processed_df['CustomerID'], errors='coerce')
        self.processed_df['Quantity'] = pd.to_numeric(self.processed_df['Quantity'])
        self.processed_df['UnitPrice'] = pd.to_numeric(self.processed_df['UnitPrice'])
        
        # Convert string columns to category for memory efficiency
        categorical_cols = ['StockCode', 'Description', 'Country']
        for col in categorical_cols:
            if col in self.processed_df.columns:
                self.processed_df[col] = self.processed_df[col].astype('category')
    
    def _remove_duplicates(self):
        """Remove duplicate transactions."""
        before_dup = len(self.processed_df)
        self.processed_df = self.processed_df.drop_duplicates()
        logger.info(f"Removed {before_dup - len(self.processed_df)} duplicate rows")
    
    def _handle_outliers(self):
        """Handle outliers in quantity and price."""
        logger.info("Handling outliers...")
        
        # Remove transactions with zero or negative unit price
        before_price = len(self.processed_df)
        self.processed_df = self.processed_df[self.processed_df['UnitPrice'] > 0]
        logger.info(f"Removed {before_price - len(self.processed_df)} rows with invalid unit price")
        
        # Handle extreme quantities (keep cancellations but remove unrealistic large quantities)
        before_qty = len(self.processed_df)
        # Remove positive quantities > 1000 (likely data entry errors)
        self.processed_df = self.processed_df[
            (self.processed_df['Quantity'] <= 1000) | 
            (self.processed_df['Quantity'] < 0)  # Keep cancellations
        ]
        logger.info(f"Removed {before_qty - len(self.processed_df)} rows with extreme quantities")
    
    def _validate_data(self):
        """Validate data integrity."""
        logger.info("Validating data integrity...")
        
        # Check for valid invoice numbers
        invalid_invoices = self.processed_df[
            ~self.processed_df['InvoiceNo'].str.match(r'^[A-Z]?\d+$', na=False)
        ]
        if len(invalid_invoices) > 0:
            logger.warning(f"Found {len(invalid_invoices)} rows with invalid invoice numbers")
        
        # Check date range
        date_range = self.processed_df['InvoiceDate'].agg(['min', 'max'])
        logger.info(f"Date range: {date_range['min']} to {date_range['max']}")
        
        # Log final statistics
        logger.info(f"Final dataset shape: {self.processed_df.shape}")
        logger.info(f"Unique customers: {self.processed_df['CustomerID'].nunique()}")
        logger.info(f"Unique products: {self.processed_df['StockCode'].nunique()}")
        logger.info(f"Date range: {self.processed_df['InvoiceDate'].min()} to {self.processed_df['InvoiceDate'].max()}")
    
    def feature_engineering(self) -> pd.DataFrame:
        """
        Create additional features for analysis.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logger.info("Creating engineered features...")
        
        if self.processed_df is None:
            raise ValueError("No processed data available. Please clean data first.")
        
        # Create revenue column
        self.processed_df['Revenue'] = self.processed_df['Quantity'] * self.processed_df['UnitPrice']
        
        # Extract time features
        self.processed_df['Year'] = self.processed_df['InvoiceDate'].dt.year
        self.processed_df['Month'] = self.processed_df['InvoiceDate'].dt.month
        self.processed_df['DayOfWeek'] = self.processed_df['InvoiceDate'].dt.dayofweek
        self.processed_df['Hour'] = self.processed_df['InvoiceDate'].dt.hour
        self.processed_df['Quarter'] = self.processed_df['InvoiceDate'].dt.quarter
        
        # Create cancellation flag
        self.processed_df['IsCancellation'] = self.processed_df['InvoiceNo'].str.startswith('C')
        
        # Create product categories (simplified categorization)
        self.processed_df['ProductCategory'] = self._categorize_products()
        
        # Calculate invoice totals
        invoice_totals = self.processed_df.groupby('InvoiceNo')['Revenue'].sum().reset_index()
        invoice_totals.columns = ['InvoiceNo', 'InvoiceTotal']
        self.processed_df = self.processed_df.merge(invoice_totals, on='InvoiceNo', how='left')
        
        logger.info("Feature engineering completed!")
        return self.processed_df
    
    def _categorize_products(self) -> pd.Series:
        """
        Categorize products based on description keywords.
        
        Returns:
            pd.Series: Product categories
        """
        categories = []
        
        for desc in self.processed_df['Description']:
            desc_lower = str(desc).lower()
            
            if any(word in desc_lower for word in ['bag', 'pouch', 'holder']):
                categories.append('Bags & Holders')
            elif any(word in desc_lower for word in ['clock', 'alarm']):
                categories.append('Clocks & Timepieces')
            elif any(word in desc_lower for word in ['light', 'lamp', 'candle']):
                categories.append('Lighting & Candles')
            elif any(word in desc_lower for word in ['kitchen', 'cup', 'plate', 'cake']):
                categories.append('Kitchen & Dining')
            elif any(word in desc_lower for word in ['decoration', 'ornament', 'bunting']):
                categories.append('Decorations')
            elif any(word in desc_lower for word in ['toy', 'game', 'play']):
                categories.append('Toys & Games')
            elif any(word in desc_lower for word in ['postage', 'manual', 'fee']):
                categories.append('Services')
            else:
                categories.append('Other')
        
        return pd.Series(categories)
    
    def save_processed_data(self, filepath: str):
        """
        Save processed data to file.
        
        Args:
            filepath (str): Path to save the processed data
        """
        if self.processed_df is None:
            raise ValueError("No processed data to save. Please process data first.")
        
        self.processed_df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dict: Summary statistics
        """
        if self.processed_df is None:
            raise ValueError("No processed data available.")
        
        summary = {
            'total_transactions': len(self.processed_df),
            'unique_customers': self.processed_df['CustomerID'].nunique(),
            'unique_products': self.processed_df['StockCode'].nunique(),
            'total_revenue': self.processed_df['Revenue'].sum(),
            'avg_order_value': self.processed_df.groupby('InvoiceNo')['Revenue'].sum().mean(),
            'date_range': {
                'start': self.processed_df['InvoiceDate'].min(),
                'end': self.processed_df['InvoiceDate'].max()
            },
            'top_countries': self.processed_df['Country'].value_counts().head().to_dict(),
            'cancellation_rate': self.processed_df['IsCancellation'].mean() * 100
        }
        
        return summary


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstration of the data preprocessing pipeline.
    """
    
    print("E-Commerce Data Preprocessing Demo")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = EcommerceDataPreprocessor()
    
    # Load data (using sample data for demo)
    data = preprocessor.load_data()
    print(f"\\nLoaded data shape: {data.shape}")
    
    # Clean data
    cleaned_data = preprocessor.clean_data()
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Feature engineering
    final_data = preprocessor.feature_engineering()
    print(f"Final data shape: {final_data.shape}")
    
    # Get summary
    summary = preprocessor.get_data_summary()
    print("\\nData Summary:")
    print(f"Total Transactions: {summary['total_transactions']:,}")
    print(f"Unique Customers: {summary['unique_customers']:,}")
    print(f"Unique Products: {summary['unique_products']:,}")
    print(f"Total Revenue: £{summary['total_revenue']:,.2f}")
    print(f"Average Order Value: £{summary['avg_order_value']:.2f}")
    print(f"Cancellation Rate: {summary['cancellation_rate']:.2f}%")
    
    print("\\nTop Countries:")
    for country, count in summary['top_countries'].items():
        print(f"  {country}: {count:,} transactions")
    
    # Save processed data
    # preprocessor.save_processed_data('data/processed/cleaned_transactions.csv')
    
    print("\\nPreprocessing pipeline completed successfully!")