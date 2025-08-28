#!/usr/bin/env python3
"""
Script to properly read FRED Excel files
"""
import pandas as pd
import numpy as np
import os

def read_fred_excel(file_path):
    """Read FRED Excel file, trying different methods to find the actual data"""
    print(f"Reading {file_path}")
    
    # Try reading all sheets
    xl_file = pd.ExcelFile(file_path)
    print(f"Available sheets: {xl_file.sheet_names}")
    
    for sheet_name in xl_file.sheet_names:
        print(f"\n--- Sheet: {sheet_name} ---")
        try:
            # Try reading the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"Shape: {df.shape}")
            print("First 10 rows:")
            print(df.head(10))
            
            # Look for rows that might contain actual data (numeric values)
            # Skip metadata rows and find where data starts
            for i in range(min(20, len(df))):  # Check first 20 rows
                try:
                    # Try to find a row with date-like first column and numeric second column
                    row = df.iloc[i]
                    if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                        # Try to parse the first column as date
                        try:
                            pd.to_datetime(str(row.iloc[0]))
                            print(f"Potential data starts at row {i}")
                            # Read from this row
                            df_data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=i)
                            print("Data from this row:")
                            print(df_data.head())
                            break
                        except:
                            continue
                except:
                    continue
                    
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
    
    print("=" * 80)

def examine_fred_files():
    xlsx_dir = "xlsx_data"
    files = [f for f in os.listdir(xlsx_dir) if f.endswith('.xlsx')]
    
    print("=== Examining FRED XLSX Files ===")
    print(f"Found {len(files)} xlsx files: {files}")
    print()
    
    for file in files:
        file_path = os.path.join(xlsx_dir, file)
        read_fred_excel(file_path)

if __name__ == "__main__":
    examine_fred_files()