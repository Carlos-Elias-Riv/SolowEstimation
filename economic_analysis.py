#!/usr/bin/env python3
"""
Economic Analysis to find alpha = 0.3192
This script analyzes the economic data to determine what model or relationship
could yield the target alpha parameter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def load_fred_data():
    """Load all FRED datasets"""
    data = {}
    
    # Load GDP data (Real GDP)
    gdp_df = pd.read_excel('xlsx_data/gdp.xlsx', sheet_name='Quarterly')
    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
    data['gdp'] = gdp_df.set_index('observation_date')['GDPC1']
    
    # Load GDP Deflator
    deflator_df = pd.read_excel('xlsx_data/deflator.xlsx', sheet_name='Quarterly')
    deflator_df['observation_date'] = pd.to_datetime(deflator_df['observation_date'])
    data['deflator'] = deflator_df.set_index('observation_date')['GDPDEF']
    
    # Load Investment data
    investment_df = pd.read_excel('xlsx_data/investment.xlsx', sheet_name='Quarterly')
    investment_df['observation_date'] = pd.to_datetime(investment_df['observation_date'])
    data['investment'] = investment_df.set_index('observation_date')['USAGFCFQDSMEI']
    
    # Load Labor Productivity Index
    productivity_df = pd.read_excel('xlsx_data/Indice_horas.xlsx', sheet_name='Quarterly')
    productivity_df['observation_date'] = pd.to_datetime(productivity_df['observation_date'])
    data['productivity'] = productivity_df.set_index('observation_date')['PRS85006031']
    
    return data

def create_combined_dataset(data):
    """Combine all datasets on common dates"""
    # Find common date range
    start_date = max([series.index.min() for series in data.values()])
    end_date = min([series.index.max() for series in data.values()])
    
    print(f"Common date range: {start_date} to {end_date}")
    
    # Create combined dataframe
    combined = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='QS'))
    
    for name, series in data.items():
        combined[name] = series.reindex(combined.index, method='ffill')
    
    # Remove any rows with missing data
    combined = combined.dropna()
    
    print(f"Combined dataset shape: {combined.shape}")
    print("Data preview:")
    print(combined.head())
    
    return combined

def analyze_potential_models(df):
    """Analyze different econometric models that could yield alpha = 0.3192"""
    
    target_alpha = 0.3192
    results = []
    
    print("=== SEARCHING FOR ALPHA = 0.3192 ===")
    print()
    
    # Create log transformations (common in economics)
    df_log = np.log(df)
    df_log.columns = [f'log_{col}' for col in df.columns]
    
    # Create growth rates (percentage change)
    df_growth = df.pct_change().dropna()
    df_growth.columns = [f'growth_{col}' for col in df.columns]
    
    # Create differences
    df_diff = df.diff().dropna()
    df_diff.columns = [f'diff_{col}' for col in df.columns]
    
    print("1. COBB-DOUGLAS PRODUCTION FUNCTION")
    print("   Y = A * L^alpha * K^beta")
    print("   log(Y) = log(A) + alpha*log(L) + beta*log(K)")
    
    # Try GDP as output, productivity as labor, investment as capital
    if len(df_log) > 10:
        try:
            y = df_log['log_gdp']
            X = df_log[['log_productivity', 'log_investment']]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            alpha_coef = model.params['log_productivity']
            
            print(f"   Alpha (labor elasticity): {alpha_coef:.4f}")
            print(f"   Target alpha: {target_alpha:.4f}")
            print(f"   Difference: {abs(alpha_coef - target_alpha):.4f}")
            
            results.append({
                'model': 'Cobb-Douglas (log_productivity)',
                'alpha': alpha_coef,
                'difference': abs(alpha_coef - target_alpha),
                'r_squared': model.rsquared,
                'pvalue': model.pvalues['log_productivity']
            })
            
            if abs(alpha_coef - target_alpha) < 0.01:
                print("   *** CLOSE MATCH FOUND! ***")
                print(model.summary())
                
        except Exception as e:
            print(f"   Error in Cobb-Douglas: {e}")
    
    print()
    print("2. SIMPLE LINEAR REGRESSIONS")
    
    # Try various simple regressions
    simple_models = [
        ('gdp', 'productivity'),
        ('gdp', 'investment'),
        ('gdp', 'deflator'),
        ('productivity', 'investment'),
        ('investment', 'deflator'),
        ('deflator', 'productivity')
    ]
    
    for y_var, x_var in simple_models:
        try:
            y = df[y_var]
            X = sm.add_constant(df[x_var])
            
            model = sm.OLS(y, X).fit()
            alpha_coef = model.params[x_var]
            
            print(f"   {y_var} ~ {x_var}: alpha = {alpha_coef:.6f}")
            
            results.append({
                'model': f'{y_var} ~ {x_var}',
                'alpha': alpha_coef,
                'difference': abs(alpha_coef - target_alpha),
                'r_squared': model.rsquared,
                'pvalue': model.pvalues[x_var]
            })
            
            if abs(alpha_coef - target_alpha) < 0.01:
                print("   *** CLOSE MATCH FOUND! ***")
                print(model.summary())
                
        except Exception as e:
            print(f"   Error in {y_var} ~ {x_var}: {e}")
    
    print()
    print("3. GROWTH RATE MODELS")
    
    if len(df_growth) > 10:
        for y_var, x_var in simple_models:
            try:
                y = df_growth[f'growth_{y_var}']
                X = sm.add_constant(df_growth[f'growth_{x_var}'])
                
                model = sm.OLS(y, X).fit()
                alpha_coef = model.params[f'growth_{x_var}']
                
                print(f"   growth_{y_var} ~ growth_{x_var}: alpha = {alpha_coef:.6f}")
                
                results.append({
                    'model': f'growth_{y_var} ~ growth_{x_var}',
                    'alpha': alpha_coef,
                    'difference': abs(alpha_coef - target_alpha),
                    'r_squared': model.rsquared,
                    'pvalue': model.pvalues[f'growth_{x_var}']
                })
                
                if abs(alpha_coef - target_alpha) < 0.01:
                    print("   *** CLOSE MATCH FOUND! ***")
                    print(model.summary())
                    
            except Exception as e:
                continue
    
    print()
    print("4. TRANSFORMED VARIABLES")
    
    # Try various transformations to get closer to target
    transformations = [
        ('sqrt', lambda x: np.sqrt(np.abs(x))),
        ('square', lambda x: x**2),
        ('reciprocal', lambda x: 1/x),
        ('log_diff', lambda x: np.log(x).diff())
    ]
    
    for trans_name, trans_func in transformations:
        for y_var, x_var in [('gdp', 'productivity'), ('gdp', 'investment')]:
            try:
                y_trans = trans_func(df[y_var])
                x_trans = trans_func(df[x_var])
                
                # Remove inf and nan
                mask = np.isfinite(y_trans) & np.isfinite(x_trans)
                y_clean = y_trans[mask]
                x_clean = x_trans[mask]
                
                if len(y_clean) > 10:
                    X = sm.add_constant(x_clean)
                    model = sm.OLS(y_clean, X).fit()
                    
                    # Get the coefficient for the transformed x variable
                    alpha_coef = model.params[1]  # Second parameter (first is constant)
                    
                    print(f"   {trans_name}({y_var}) ~ {trans_name}({x_var}): alpha = {alpha_coef:.6f}")
                    
                    results.append({
                        'model': f'{trans_name}({y_var}) ~ {trans_name}({x_var})',
                        'alpha': alpha_coef,
                        'difference': abs(alpha_coef - target_alpha),
                        'r_squared': model.rsquared,
                        'pvalue': model.pvalues[1]
                    })
                    
                    if abs(alpha_coef - target_alpha) < 0.01:
                        print("   *** CLOSE MATCH FOUND! ***")
                        print(model.summary())
                        
            except Exception as e:
                continue
    
    return results

def find_best_matches(results, target_alpha=0.3192):
    """Find the models closest to target alpha"""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('difference')
    
    print("\n=== TOP 10 CLOSEST MATCHES TO ALPHA = 0.3192 ===")
    print(results_df.head(10).to_string(index=False))
    
    return results_df

def main():
    print("Loading economic data...")
    data = load_fred_data()
    
    print("\nCreating combined dataset...")
    combined_df = create_combined_dataset(data)
    
    print("\nAnalyzing models to find alpha = 0.3192...")
    results = analyze_potential_models(combined_df)
    
    print("\nFinding best matches...")
    best_matches = find_best_matches(results)
    
    # Save results
    best_matches.to_csv('alpha_search_results.csv', index=False)
    print("\nResults saved to 'alpha_search_results.csv'")
    
    return best_matches, combined_df

if __name__ == "__main__":
    best_matches, data = main()