#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_realistic_water_pump_dataset(n_samples=1000):
    """Create a realistic sample dataset with real data quality issues"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic values for Tanzania
    regions = ['Dodoma', 'Arusha', 'Kilimanjaro', 'Tanga', 'Morogoro', 'Pwani', 'Dar es Salaam', 
               'Lindi', 'Mtwara', 'Ruvuma', 'Iringa', 'Mbeya', 'Singida', 'Tabora', 'Rukwa', 
               'Kigoma', 'Shinyanga', 'Kagera', 'Mwanza', 'Mara', 'Manyara', 'Njombe', 'Katavi', 
               'Simiyu', 'Geita', 'Songwe']
    
    basins = ['Lake Victoria', 'Pangani', 'Internal', 'Rufiji', 'Ruvuma / Southern Coast', 
              'Lake Nyasa', 'Lake Rukwa', 'Lake Tanganyika', 'Wami / Ruvu']
    
    funders = ['Government Of Tanzania', 'Danida', 'Hesawa', 'Rwssp', 'World Bank', 
               'Kkkt', 'Tasaf', 'Dwe', 'Wsdp', 'Tcrs', 'Unicef', 'Private individual', 
               'World Vision', 'Oxfam', 'District Council', 'Rural Water Supply And Sanitation Programme',
               'Community', 'Other']
    
    water_quality = ['soft', 'salty', 'milky', 'colored', 'fluoride', 'fluoride abandoned']
    quantity = ['enough', 'insufficient', 'dry', 'seasonal']
    status_groups = ['functional', 'functional needs repair', 'non functional']
    
    # Create base data
    data = []
    
    for i in range(n_samples):
        # Add realistic missing values and data quality issues
        record = {
            'id': 10000 + i,
            'longitude': np.random.uniform(29.0, 40.5) if np.random.random() > 0.02 else np.nan,  # 2% missing
            'latitude': np.random.uniform(-11.7, -0.95) if np.random.random() > 0.02 else np.nan,  # 2% missing
            'date_recorded': (datetime(2011, 1, 1) + timedelta(days=np.random.randint(0, 1000))).strftime('%Y-%m-%d'),
            'funder': np.random.choice(funders) if np.random.random() > 0.15 else np.nan,  # 15% missing
            'gps_height': np.random.randint(0, 2500) if np.random.random() > 0.05 else np.nan,  # 5% missing
            'installer': np.random.choice(funders) if np.random.random() > 0.20 else np.nan,  # 20% missing
            'wpt_name': f'Water_Point_{i}' if np.random.random() > 0.05 else np.nan,  # 5% missing
            'num_private': np.random.randint(0, 10) if np.random.random() > 0.30 else np.nan,  # 30% missing
            'basin': np.random.choice(basins),
            'subvillage': f'Subvillage_{np.random.randint(1, 100)}' if np.random.random() > 0.10 else np.nan,  # 10% missing
            'region': np.random.choice(regions),
            'region_code': np.random.randint(1, 26),
            'district_code': np.random.randint(1, 100),
            'lga': f'LGA_{np.random.randint(1, 50)}',
            'ward': f'Ward_{np.random.randint(1, 200)}',
            'recorded_by': 'GeoData Consultants Ltd',
            'scheme_management': np.random.choice(['VWC', 'WUA', 'Water authority', 'Private operator', 'Other']) if np.random.random() > 0.25 else np.nan,
            'scheme_name': f'Scheme_{np.random.randint(1, 100)}' if np.random.random() > 0.60 else np.nan,  # 60% missing
            'permit': np.random.choice([True, False]) if np.random.random() > 0.40 else np.nan,  # 40% missing
            'construction_year': np.random.randint(1960, 2013) if np.random.random() > 0.35 else np.nan,  # 35% missing
            'extraction_type': np.random.choice(['gravity', 'handpump', 'submersible', 'motorpump', 'other']) if np.random.random() > 0.05 else np.nan,
            'extraction_type_group': np.random.choice(['gravity', 'handpump', 'motorpump', 'other']) if np.random.random() > 0.05 else np.nan,
            'extraction_type_class': np.random.choice(['gravity', 'handpump', 'motorpump', 'other']) if np.random.random() > 0.05 else np.nan,
            'management': np.random.choice(['vwc', 'wua', 'water authority', 'private operator', 'other']) if np.random.random() > 0.08 else np.nan,
            'management_group': np.random.choice(['user-group', 'commercial', 'parastatal', 'other']) if np.random.random() > 0.08 else np.nan,
            'payment': np.random.choice(['pay per bucket', 'never pay', 'pay monthly', 'pay annually', 'other']) if np.random.random() > 0.10 else np.nan,
            'payment_type': np.random.choice(['pay per bucket', 'never pay', 'pay monthly', 'pay annually', 'other']) if np.random.random() > 0.10 else np.nan,
            'water_quality': np.random.choice(water_quality) if np.random.random() > 0.08 else np.nan,
            'quality_group': np.random.choice(['good', 'salty', 'milky', 'colored', 'fluoride']) if np.random.random() > 0.08 else np.nan,
            'quantity': np.random.choice(quantity) if np.random.random() > 0.05 else np.nan,
            'quantity_group': np.random.choice(['enough', 'insufficient', 'dry']) if np.random.random() > 0.05 else np.nan,
            'source': np.random.choice(['spring', 'borehole', 'well', 'river', 'lake', 'other']) if np.random.random() > 0.03 else np.nan,
            'source_type': np.random.choice(['spring', 'borehole', 'shallow well', 'river', 'lake', 'other']) if np.random.random() > 0.03 else np.nan,
            'source_class': np.random.choice(['groundwater', 'surface']) if np.random.random() > 0.03 else np.nan,
            'waterpoint_type': np.random.choice(['communal standpipe', 'hand pump', 'other', 'improved spring', 'cattle trough']) if np.random.random() > 0.03 else np.nan,
            'waterpoint_type_group': np.random.choice(['communal standpipe', 'hand pump', 'other', 'improved spring', 'cattle trough']) if np.random.random() > 0.03 else np.nan,
        }
        
        # Add realistic status_group with probabilities based on other factors
        # Non-functional pumps are more likely with:
        # - Older construction year
        # - Certain extraction types
        # - Poor water quality
        # - Insufficient quantity
        
        prob_functional = 0.54  # Base probability
        
        # Adjust based on construction year
        if not pd.isna(record['construction_year']):
            if record['construction_year'] < 1980:
                prob_functional -= 0.15
            elif record['construction_year'] < 1990:
                prob_functional -= 0.10
            elif record['construction_year'] < 2000:
                prob_functional -= 0.05
        
        # Adjust based on water quality
        if record['water_quality'] in ['salty', 'fluoride', 'colored']:
            prob_functional -= 0.10
        
        # Adjust based on quantity
        if record['quantity'] in ['dry', 'insufficient']:
            prob_functional -= 0.15
        
        # Ensure probabilities are valid
        prob_functional = max(0.1, min(0.8, prob_functional))
        prob_needs_repair = min(0.3, (1 - prob_functional) * 0.4)
        prob_non_functional = 1 - prob_functional - prob_needs_repair
        
        record['status_group'] = np.random.choice(
            status_groups,
            p=[prob_functional, prob_needs_repair, prob_non_functional]
        )
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some outliers for realistic data
    # Outliers in GPS coordinates
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'longitude'] = np.random.uniform(-10, 10, size=len(outlier_indices))
    df.loc[outlier_indices, 'latitude'] = np.random.uniform(-20, 5, size=len(outlier_indices))
    
    # Outliers in GPS height
    outlier_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[outlier_indices, 'gps_height'] = np.random.uniform(5000, 10000, size=len(outlier_indices))
    
    # Add some duplicate rows (realistic data issue)
    n_duplicates = int(0.02 * len(df))  # 2% duplicates
    duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    duplicates['id'] = range(20000, 20000 + len(duplicates))  # Change ID to avoid exact duplicates
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

if __name__ == "__main__":
    # Create realistic sample dataset
    print("Creating realistic sample dataset...")
    df = create_realistic_water_pump_dataset(1000)
    
    # Save to sample_data directory
    os.makedirs("sample_data", exist_ok=True)
    df.to_csv("sample_data/sample_water_pumps.csv", index=False)
    
    print(f"Dataset created with {len(df)} records")
    print(f"Missing values per column:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"\nStatus group distribution:")
    print(df['status_group'].value_counts())
    print(f"\nDataset saved to sample_data/sample_water_pumps.csv")