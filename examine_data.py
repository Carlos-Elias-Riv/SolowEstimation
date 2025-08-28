#!/usr/bin/env python3
"""
Script to examine the xlsx files and understand their structure
"""
import pandas as pd
import numpy as np
import os

def examine_xlsx_files():
    xlsx_dir = "xlsx_data"
    files = [f for f in os.listdir(xlsx_dir) if f.endswith('.xlsx')]
    
    print("=== Examining XLSX Files ===")
    print(f"Found {len(files)} xlsx files: {files}")
    print()
    
    for file in files:
        file_path = os.path.join(xlsx_dir, file)
        print(f"--- {file} ---")
        
        try:
            # Try to read the file and examine its structure
            df = pd.read_excel(file_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("First few rows:")
            print(df.head())
            print()
            print("Data types:")
            print(df.dtypes)
            print()
            print("Summary statistics:")
            print(df.describe())
            print("=" * 60)
            print()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            print()

if __name__ == "__main__":
    examine_xlsx_files()