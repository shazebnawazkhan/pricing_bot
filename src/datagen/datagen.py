import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import concurrent.futures
import string

# --- CONFIGURATION ---
NUM_ACCOUNTS = 200
NUM_PRODUCTS = 5000
TOTAL_ROWS = 1_000_000  # Set to several million as needed
NUM_THREADS = 8         # Adjust based on your CPU cores
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 12, 31)

def generate_master_data():
    """Generates static Account and Product lookup tables."""
    
    # 1. Accounts Master
    regions = {
        'US': ['USA', 'Canada'],
        'Europe': ['UK', 'Germany', 'France', 'Switzerland'],
        'Asia': ['Singapore', 'Japan', 'India', 'Hong Kong']
    }
    
    acc_data = []
    for i in range(NUM_ACCOUNTS):
        reg = random.choice(list(regions.keys()))
        country = random.choice(regions[reg])
        acc_data.append({
            'account_id': f'ACC_{i:04d}',
            'region': reg,
            'country': country,
            'client_name': f'Client_{''.join(random.choices(string.ascii_uppercase, k=5))}',
            'parent_name': f'Parent_Corp_{random.randint(1, 50)}',
            'asset_size': np.random.uniform(1e6, 1e11) # $1M to $100B
        })
    df_accounts = pd.DataFrame(acc_data)

    # 2. Products Master
    prod_data = []
    for i in range(NUM_PRODUCTS):
        prod_data.append({
            'product_code': f'PRD_{i:05d}',
            'product_family': random.choice(['Risk', 'Lending', 'Payments', 'Wealth']),
            'product_group': f'Group_{random.randint(1, 10)}',
            'product_sub_group': f'SubGroup_{random.randint(1, 50)}',
            'l1': 'SaaS_Banking', 'l2': 'Digital_Core', 'l3': 'Cloud_Services',
            'l4': 'Standard', 'l5': 'Tier_1', 'l6': 'Active',
            'list_price': np.random.uniform(500, 20000)
        })
    df_products = pd.DataFrame(prod_data)
    
    return df_accounts, df_products

def generate_chunk(chunk_size, accounts, products, start_date, end_date):
    """Generates a portion of the transaction data."""
    
    # Randomly sample from masters to ensure even distribution
    acc_sample = accounts.sample(chunk_size, replace=True).reset_index(drop=True)
    prod_sample = products.sample(chunk_size, replace=True).reset_index(drop=True)
    
    # Generate random dates
    delta_days = (end_date - start_date).days
    random_days = np.random.randint(0, delta_days, size=chunk_size)
    sale_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Generate transactional attributes
    terms = np.random.choice([12, 24, 36, 60], size=chunk_size) # Months
    quantities = np.random.randint(1, 100, size=chunk_size)
    
    data = {
        'opportunity_id': [f'OPP_{random.randint(100000, 999999)}' for _ in range(chunk_size)],
        'opportunity_stage': np.random.choice(['Closed Won', 'Closed Lost', 'Negotiation'], size=chunk_size, p=[0.6, 0.2, 0.2]),
        'opportunity_name': [f'Deal_{i}' for i in range(chunk_size)],
        'term': terms,
        'quantity': quantities,
        'saas_migration': np.random.choice([True, False], size=chunk_size),
        'sale_date': sale_dates,
        'sales_year': [d.year for d in sale_dates],
    }
    
    # Combine everything
    df_chunk = pd.concat([acc_sample, prod_sample, pd.DataFrame(data)], axis=1)
    
    # Derived calculations
    df_chunk['sales_amount'] = df_chunk['list_price'] * df_chunk['quantity'] * (1 - np.random.uniform(0, 0.3, chunk_size))
    df_chunk['contract_start_date'] = df_chunk['sale_date'] + pd.to_timedelta(np.random.randint(7, 30, chunk_size), unit='D')
    
    # Function to calculate end date based on term
    df_chunk['contract_end_date'] = df_chunk.apply(
        lambda x: x['contract_start_date'] + pd.DateOffset(months=int(x['term'])), axis=1
    )
    
    return df_chunk

def main():
    print(f"Starting data generation for {TOTAL_ROWS} rows...")
    df_accounts, df_products = generate_master_data()
    
    chunk_size = TOTAL_ROWS // NUM_THREADS
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for _ in range(NUM_THREADS):
            futures.append(executor.submit(
                generate_chunk, chunk_size, df_accounts, df_products, START_DATE, END_DATE
            ))
            
    results = [f.result() for f in futures]
    final_df = pd.concat(results, ignore_index=True)
    
    print(f"Successfully generated {len(final_df)} rows.")
    print(final_df.head())
    
    # Optional: Save to CSV
    final_df.to_csv('saas_transactions.csv', index=False)
    return final_df

if __name__ == "__main__":
    df = main()