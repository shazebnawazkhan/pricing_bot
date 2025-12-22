import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import string

# --- CONFIGURATION ---
NUM_ACCOUNTS = 200
NUM_PRODUCTS = 5000
TOTAL_ROWS = 1_000_000  # Set to 5,000_000+ for "several million"
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 12, 31)

# --- MASTER DATA GENERATION ---

def generate_account_master(n):
    regions = {
        'Americas': ['US', 'Canada', 'Brazil'],
        'Europe': ['UK', 'Germany', 'France', 'Switzerland'],
        'Asia': ['Singapore', 'Japan', 'Hong Kong', 'India']
    }
    
    data = []
    for i in range(n):
        reg = random.choice(list(regions.keys()))
        cntry = random.choice(regions[reg])
        acc_id = f"ACC-{10000 + i}"
        data.append({
            'account_id': acc_id,
            'region': reg,
            'country': cntry,
            'client_name': f"Client {string.ascii_uppercase[i % 26]}{i}",
            'parent_name': f"Parent Corp {random.randint(1, 50)}",
            'asset_size': random.choice(['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']),
            'account_manager': f"Manager {random.randint(1, 20)}"
        })
    return pd.DataFrame(data)

def generate_product_master(n):
    # Moody's Analytics style hierarchies
    l1_categories = ['Risk Management', 'Commercial Lending', 'Financial & Regulatory', 'ESG & Climate']
    families = ['CreditLens', 'RiskCalc', 'ImpairmentStudio', 'CreditView', 'MA Data Services']
    
    data = []
    for i in range(n):
        l1 = random.choice(l1_categories)
        prod_code = f"MA-{20000 + i}"
        data.append({
            'product_code': prod_code,
            'l1': l1,
            'l2': f"{l1} - Level 2",
            'l3': f"Sub-Category {random.randint(1, 10)}",
            'l4': f"Function Group {random.randint(1, 5)}",
            'l5': "Standard SaaS",
            'l6': "Tiered License",
            'product_family': random.choice(families),
            'product_group': f"Group-{random.randint(1, 15)}",
            'product_sub_group': f"SubGroup-{random.randint(1, 30)}",
            'list_price': round(random.uniform(5000, 150000), 2)
        })
    return pd.DataFrame(data)

# --- WORKER FUNCTION FOR MULTIPROCESSING ---

def generate_chunk(chunk_size, acc_df, prod_df):
    """Generates a chunk of transaction data."""
    stages = ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
    sale_types = ['New Business', 'Renewal', 'Upsell', 'Cross-sell']
    
    # Pre-calculate date range
    delta_days = (END_DATE - START_DATE).days
    
    # Efficiently sample indices
    acc_indices = np.random.randint(0, len(acc_df), chunk_size)
    prod_indices = np.random.randint(0, len(prod_df), chunk_size)
    
    # Data creation
    rows = []
    for i in range(chunk_size):
        acc = acc_df.iloc[acc_indices[i]]
        prod = prod_df.iloc[prod_indices[i]]
        
        # Salesforce-style Opportunity
        opp_id = f"006{random.randint(1000000, 9999999)}"
        sale_date = START_DATE + timedelta(days=random.randint(0, delta_days))
        term_months = random.choice([12, 24, 36])
        quantity = random.randint(1, 10)
        
        # Financial logic
        sales_amt = prod['list_price'] * quantity * random.uniform(0.8, 1.1) # Discount/Premium
        
        row = {
            **acc.to_dict(),
            **prod.to_dict(),
            'opportunity_id': opp_id,
            'opportunity_stage': random.choice(stages),
            'opportunity_name': f"Opp - {acc['client_name']} - {prod['product_family']}",
            'contract_start_date': sale_date,
            'contract_end_date': sale_date + timedelta(days=30 * term_months),
            'term': term_months,
            'sales_amount': round(sales_amt, 2),
            'quantity': quantity,
            'saas_migration': random.choice([True, False]),
            'sale_date': sale_date,
            'sales_year': sale_date.year,
            'probability': random.randint(0, 100),
            'sales_rep': f"Rep {random.randint(1, 100)}",
            'sale_type': random.choice(sale_types)
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"Starting Data Generation for {TOTAL_ROWS} rows...")
    
    # 1. Setup Master Data
    acc_master = generate_account_master(NUM_ACCOUNTS)
    prod_master = generate_product_master(NUM_PRODUCTS)
    
    # 2. Split tasks for multiprocessing
    num_cores = cpu_count()
    chunk_size = TOTAL_ROWS // num_cores
    
    print(f"Using {num_cores} cores with chunk size {chunk_size}...")
    
    with Pool(processes=num_cores) as pool:
        # Pass static data to each worker
        results = [
            pool.apply_async(generate_chunk, args=(chunk_size, acc_master, prod_master))
            for _ in range(num_cores)
        ]
        
        # Combine results
        final_df = pd.concat([r.get() for r in results], ignore_index=True)

    print(f"Generation Complete. Total Rows: {len(final_df)}")
    
    # Export or Analysis
    final_df.to_csv("saas_pricing_data3.csv", index=False)
    print(final_df.head())