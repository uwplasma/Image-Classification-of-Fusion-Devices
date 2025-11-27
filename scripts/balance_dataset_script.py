import pandas as pd
import numpy as np
from pathlib import Path
import os


'''
Notes
Want to balance the dataset since right now there is a off balance of the nfp.
   * Note: In reality, NFP 3 and 4 are most common that's why we'll have more distribution of that
           NFP 6 is hardly ever seen, so that's why theres none of those
           NFP 1 is also hardly ever seen, but we included a couple of them for the model
   * When we train the model with the imbalanced dataset, that causes the model 
     to do worse with the less balanced stuff 
   
Balance Dataset
---------------------    
1. Create a csv subset of XGStesls (have it the same format of XGStels but 
    just have the 5000 rows below)
    
    * Have this distrubution of nfps:
     nfp    num rows
     3      2120
     4      1450
     2      800
     5      550
     1      80

'''


def balance_dataset(input_csv_path, output_csv_path):
    """
    Create a balanced subset of the dataset with specified NFP distribution.
    
    Args:
        input_csv_path: Path to the original XGStels dataset
        output_csv_path: Path to save the balanced subset
    """
    target_distribution = {
        3: 2120,
        4: 1450,
        2: 800,
        5: 550,
        1: 80
    }
    
    print(f"Reading dataset from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    print("\nOriginal NFP distribution:")
    print(df['nfp'].value_counts().sort_index())
    
    balanced_frames = []
    
    for nfp_value, target_count in target_distribution.items():
        nfp_rows = df[df['nfp'] == nfp_value]
        
        available_count = len(nfp_rows)
        print(f"\nNFP {nfp_value}: Available={available_count}, Target={target_count}")
        
        if available_count >= target_count:
            sampled_rows = nfp_rows.sample(n=target_count, random_state=42)
        else:
            print(f"  Warning: Only {available_count} samples available for NFP {nfp_value}, using all")
            sampled_rows = nfp_rows
        
        balanced_frames.append(sampled_rows)
    
    balanced_df = pd.concat(balanced_frames, ignore_index=True)
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nBalanced NFP distribution:")
    print(balanced_df['nfp'].value_counts().sort_index())
    print(f"\nTotal rows: {len(balanced_df)}")
    
    balanced_df.to_csv(output_csv_path, index=False)
    print(f"\nBalanced dataset saved to {output_csv_path}")
    
    return balanced_df

if __name__ == "__main__":
    base_dir = Path("/home/exouser/Public/Image-Classification-Updated")
    
    print("Searching for XGStels.csv file...")
    csv_files = list(base_dir.rglob("XGStels.csv"))
    
    if csv_files:
        input_path = csv_files[0]
        print(f"Found: {input_path}")
    else:
        print("\nXGStels.csv not found. Please enter the correct path:")
        print("Available CSV files in project:")
        for csv_file in base_dir.rglob("*.csv"):
            print(f"  - {csv_file}")
        exit(1)
    
    output_path = input_path.parent / "XGStels_balanced.csv"
    
    balanced_df = balance_dataset(input_path, output_path)