#!/usr/bin/env python3
"""
Solow Growth Model Estimation - Homework Solution
Objective: Estimate α = 0.3192 in the production function Yt = A * Kt^α * Lt^(1-α)

Following the exact methodology described in caso1.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare all datasets according to homework instructions"""
    
    print("=== LOADING AND PREPARING DATA ===")
    
    # 1. Load Labor Hours Index (Indicehoras.xlsx) - This will be our base dataset
    print("Loading labor hours index...")
    labor_df = pd.read_excel('xlsx_data/Indice_horas.xlsx', sheet_name='Quarterly')
    labor_df['observation_date'] = pd.to_datetime(labor_df['observation_date'])
    labor_df = labor_df.set_index('observation_date')['PRS85006031']
    print(f"Labor data: {len(labor_df)} observations from {labor_df.index.min()} to {labor_df.index.max()}")
    
    # 2. Load GDP data (Real GDP)
    print("Loading GDP data...")
    gdp_df = pd.read_excel('xlsx_data/gdp.xlsx', sheet_name='Quarterly')
    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
    gdp_df = gdp_df.set_index('observation_date')['GDPC1']
    print(f"GDP data: {len(gdp_df)} observations from {gdp_df.index.min()} to {gdp_df.index.max()}")
    
    # 3. Load GDP Deflator
    print("Loading GDP deflator...")
    deflator_df = pd.read_excel('xlsx_data/deflator.xlsx', sheet_name='Quarterly')
    deflator_df['observation_date'] = pd.to_datetime(deflator_df['observation_date'])
    deflator_df = deflator_df.set_index('observation_date')['GDPDEF']
    print(f"Deflator data: {len(deflator_df)} observations from {deflator_df.index.min()} to {deflator_df.index.max()}")
    
    # 4. Load Investment data (needs special treatment)
    print("Loading and treating investment data...")
    investment_df = pd.read_excel('xlsx_data/investment.xlsx', sheet_name='Quarterly')
    investment_df['observation_date'] = pd.to_datetime(investment_df['observation_date'])
    investment_df = investment_df.set_index('observation_date')['USAGFCFQDSMEI']
    print(f"Investment data: {len(investment_df)} observations from {investment_df.index.min()} to {investment_df.index.max()}")
    
    # 5. Create combined dataset using labor hours as base
    print("\nCreating combined dataset...")
    
    # Filter data from Q1 1960 to Q3 2023 (as per homework instructions)
    start_date = '1960-01-01'
    end_date = '2023-07-01'  # Q3 2023
    
    # Use labor hours index as base and add other variables
    combined = pd.DataFrame(index=labor_df.index)
    combined['labor_hours_index'] = labor_df
    
    # Add other variables, aligning with labor hours dates
    combined['gdp'] = gdp_df.reindex(combined.index)
    combined['deflator'] = deflator_df.reindex(combined.index)
    combined['investment_nominal'] = investment_df.reindex(combined.index)
    
    # Filter to the specified period
    combined = combined[(combined.index >= start_date) & (combined.index <= end_date)]
    
    # Remove rows with missing data
    print(f"Before removing missing data: {len(combined)} observations")
    combined = combined.dropna()
    print(f"After removing missing data: {len(combined)} observations")
    
    # 6. Treat investment data according to homework instructions
    print("\nTreating investment data...")
    # Investment needs to be: multiplied by 4, converted to billions, deflated
    # Note: The data is already in billions, so we multiply by 4 and deflate
    combined['investment_real'] = (combined['investment_nominal'] * 4) / combined['deflator']
    
    print(f"Final dataset: {len(combined)} observations from {combined.index.min()} to {combined.index.max()}")
    
    return combined

def construct_capital_series(df):
    """Construct capital series using perpetual inventory method"""
    
    print("\n=== CONSTRUCTING CAPITAL SERIES ===")
    
    # Parameters
    delta = 0.05  # Depreciation rate
    
    # Initial capital: K1960Q1 = 3 * Y1960Q1
    initial_gdp = df['gdp'].iloc[0]
    initial_capital = 3 * initial_gdp
    
    print(f"Initial GDP (1960Q1): {initial_gdp:.2f}")
    print(f"Initial Capital (K1960Q1): {initial_capital:.2f}")
    
    # Initialize capital series
    capital_series = np.zeros(len(df))
    capital_series[0] = initial_capital
    
    # Apply perpetual inventory method: Kt+1 = Kt(1-δ) + It
    for t in range(1, len(df)):
        capital_series[t] = capital_series[t-1] * (1 - delta) + df['investment_real'].iloc[t-1]
    
    df['capital'] = capital_series
    
    # Remove first 4 observations as instructed (estimation starts from 1961Q1)
    print(f"Before removing first 4 observations: {len(df)} observations")
    df_estimation = df.iloc[4:].copy()
    print(f"After removing first 4 observations: {len(df_estimation)} observations")
    print(f"Estimation period: {df_estimation.index.min()} to {df_estimation.index.max()}")
    
    return df_estimation

def prepare_growth_rates(df):
    """Prepare growth rates according to homework methodology"""
    
    print("\n=== PREPARING GROWTH RATES ===")
    
    # Calculate labor growth from the index
    # Lt represents quarterly growth vs same quarter previous year
    # The index is already in this format, so we use it directly as growth rates
    df['labor_growth'] = df['labor_hours_index'] / 100  # Convert from percentage to decimal
    
    # Calculate GDP and Capital growth rates using the specified formula:
    # GrowthX,t = ln(Xt) - ln(Xt-1) = Xt/Xt-1 - 1
    
    # GDP growth
    df['gdp_growth'] = np.log(df['gdp']) - np.log(df['gdp'].shift(1))
    
    # Capital growth  
    df['capital_growth'] = np.log(df['capital']) - np.log(df['capital'].shift(1))
    
    # Remove first observation due to lagged calculation
    df_growth = df.iloc[1:].copy()
    
    print(f"Growth rates dataset: {len(df_growth)} observations")
    print(f"Period: {df_growth.index.min()} to {df_growth.index.max()}")
    
    # Display some statistics
    print("\nGrowth rate statistics:")
    print(f"GDP growth: mean={df_growth['gdp_growth'].mean():.4f}, std={df_growth['gdp_growth'].std():.4f}")
    print(f"Capital growth: mean={df_growth['capital_growth'].mean():.4f}, std={df_growth['capital_growth'].std():.4f}")
    print(f"Labor growth: mean={df_growth['labor_growth'].mean():.4f}, std={df_growth['labor_growth'].std():.4f}")
    
    return df_growth

def estimate_alpha(df):
    """Estimate α using the growth rate methodology"""
    
    print("\n=== ESTIMATING ALPHA ===")
    
    # Transform the production function to growth rates
    # Yt = A * Kt^α * Lt^(1-α)
    # Taking logs: ln(Yt) = ln(At) + α*ln(Kt) + (1-α)*ln(Lt)
    # Taking differences: Δln(Yt) = Δln(At) + α*Δln(Kt) + (1-α)*Δln(Lt)
    # Rearranging: Δln(Yt) - Δln(Lt) = Δln(At) + α*(Δln(Kt) - Δln(Lt))
    
    # Create the dependent and independent variables
    y = df['gdp_growth'] - df['labor_growth']  # Δln(Y) - Δln(L)
    x = df['capital_growth'] - df['labor_growth']  # Δln(K) - Δln(L)
    
    # Remove any remaining missing values
    mask = ~(np.isnan(y) | np.isnan(x) | np.isinf(y) | np.isinf(x))
    y_clean = y[mask]
    x_clean = x[mask]
    
    print(f"Clean data for estimation: {len(y_clean)} observations")
    
    # Add constant term for the regression
    X_with_constant = sm.add_constant(x_clean)
    
    # Run OLS regression
    model = sm.OLS(y_clean, X_with_constant).fit()
    
    # Extract α coefficient
    alpha_estimated = model.params.iloc[1]  # Second parameter (first is constant)
    
    print(f"\n=== ESTIMATION RESULTS ===")
    print(f"Estimated α: {alpha_estimated:.6f}")
    print(f"Target α: 0.3192")
    print(f"Difference: {abs(alpha_estimated - 0.3192):.6f}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"P-value for α: {model.pvalues.iloc[1]:.2e}")
    print(f"95% Confidence Interval: [{model.conf_int().iloc[1,0]:.4f}, {model.conf_int().iloc[1,1]:.4f}]")
    
    if abs(alpha_estimated - 0.3192) < 0.001:
        print("\n*** SUCCESS! Target α = 0.3192 achieved! ***")
    else:
        print(f"\n*** Close to target. Difference: {abs(alpha_estimated - 0.3192):.6f} ***")
    
    print("\n=== FULL MODEL SUMMARY ===")
    print(model.summary())
    
    return model, alpha_estimated

def main():
    """Main function to execute the complete Solow estimation"""
    
    print("SOLOW GROWTH MODEL ESTIMATION")
    print("Objective: Estimate α = 0.3192")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Construct capital series
    df_with_capital = construct_capital_series(df)
    
    # Step 3: Prepare growth rates
    df_growth = prepare_growth_rates(df_with_capital)
    
    # Step 4: Estimate α
    model, alpha = estimate_alpha(df_growth)
    
    # Step 5: Save results
    print(f"\n=== SAVING RESULTS ===")
    
    # Save the final dataset
    df_growth.to_csv('solow_estimation_data.csv')
    print("Data saved to: solow_estimation_data.csv")
    
    # Save model results
    with open('solow_estimation_results.txt', 'w') as f:
        f.write("SOLOW GROWTH MODEL ESTIMATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Estimated α: {alpha:.6f}\n")
        f.write(f"Target α: 0.3192\n")
        f.write(f"Difference: {abs(alpha - 0.3192):.6f}\n")
        f.write(f"R-squared: {model.rsquared:.4f}\n")
        f.write(f"Number of observations: {len(model.fittedvalues)}\n\n")
        f.write("FULL MODEL SUMMARY:\n")
        f.write(str(model.summary()))
    
    print("Results saved to: solow_estimation_results.txt")
    
    return df_growth, model, alpha

if __name__ == "__main__":
    final_data, final_model, final_alpha = main()