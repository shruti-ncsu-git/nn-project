import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(merged_csv):
    # 1. Load the merged dataset
    df = pd.read_csv(merged_csv)
    
    # Convert the 'Date' column to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Basic Information
    print("==== DataFrame Info ====")
    print(df.info())
    
    print("\n==== First 5 Rows ====")
    print(df.head())
    
    print("\n==== Summary Statistics ====")
    print(df.describe())
    
    # 3. Check for missing values
    missing = df.isnull().sum()
    print("\n==== Missing Values Per Column ====")
    print(missing[missing > 0])
    
    # 4. Plot Time Series of Daily Traffic Volume (QT_VOLUME_24HOUR)
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['QT_VOLUME_24HOUR'], marker='o', linestyle='-')
    plt.xlabel("Date")
    plt.ylabel("Daily Traffic Volume")
    plt.title("Daily Traffic Volume over 2023 for Site 100")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 5. Histogram of Daily Traffic Volume
    plt.figure(figsize=(8,6))
    sns.histplot(df['QT_VOLUME_24HOUR'], bins=30, kde=True)
    plt.xlabel("Daily Traffic Volume")
    plt.title("Distribution of Daily Traffic Volume")
    plt.tight_layout()
    plt.show()
    
    # 6. Plot Histograms for Weather Variables (if available)
    # Typical Meteostat daily variables: tavg, tmin, tmax, prcp, wspd, etc.
    weather_vars = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd']
    for var in weather_vars:
        if var in df.columns:
            plt.figure(figsize=(8,6))
            sns.histplot(df[var].dropna(), bins=30, kde=True)
            plt.xlabel(var)
            plt.title(f"Distribution of {var.upper()}")
            plt.tight_layout()
            plt.show()
    
    # 7. Correlation Matrix among Numeric Columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14,10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to the merged CSV file (adjust as needed)
    merged_csv = "/Users/shrutichintalapati/Documents/nn project/datafinal.csv"
    perform_eda(merged_csv)
