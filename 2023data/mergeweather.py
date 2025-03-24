import os
import pandas as pd
import datetime
from meteostat import Point, Daily

def load_traffic_data(traffic_csv):
    """
    Loads traffic data from the given CSV file and converts the
    'QT_INTERVAL_COUNT' column to a proper Date column.
    """
    df = pd.read_csv(traffic_csv)
    # Convert the QT_INTERVAL_COUNT column (assumed to hold datetime strings)
    # to a date object, then store as a new 'Date' column.
    df['Date'] = pd.to_datetime(df['QT_INTERVAL_COUNT']).dt.date
    return df

def fetch_weather_data(start, end, lat, lon):
    """
    Fetches daily weather data for the given date range and location.
    Returns a DataFrame with a 'Date' column.
    """
    location = Point(lat, lon)
    weather_data = Daily(location, start, end).fetch()
    weather_data.reset_index(inplace=True)
    weather_data.rename(columns={'time': 'Date'}, inplace=True)
    weather_data['Date'] = pd.to_datetime(weather_data['Date']).dt.date
    return weather_data

def merge_traffic_weather(traffic_df, weather_df):
    """
    Merges traffic and weather data on the 'Date' column.
    """
    merged_df = pd.merge(traffic_df, weather_df, on='Date', how='left')
    return merged_df

if __name__ == "__main__":
    # Define file paths
    project_dir = "/Users/shrutichintalapati/Documents/nn project"
    traffic_csv = os.path.join(project_dir, "site100_all.csv")
    merged_csv = os.path.join(project_dir, "site100_merged_weather.csv")
    
    # 1. Load traffic data from site100_all.csv
    print("Loading traffic data...")
    traffic_df = load_traffic_data(traffic_csv)
    print(f"Traffic data shape: {traffic_df.shape}")
    print(traffic_df.head())
    
    # 2. Define date range for weather data (for 2023)
    start_date = datetime.datetime(2023, 1, 1)
    end_date   = datetime.datetime(2023, 12, 31)
    
    # 3. Fetch weather data for Victoria (Melbourne as example)
    print("Fetching weather data from Meteostat...")
    # Coordinates for Melbourne, Victoria
    lat, lon = -37.8136, 144.9631
    weather_df = fetch_weather_data(start_date, end_date, lat, lon)
    print("Weather data shape:", weather_df.shape)
    print(weather_df.head())
    
    # 4. Merge traffic and weather data on 'Date'
    print("Merging traffic and weather data...")
    merged_df = merge_traffic_weather(traffic_df, weather_df)
    print("Merged DataFrame shape:", merged_df.shape)
    print(merged_df.head())
    
    # 5. Write merged data to CSV
    merged_df.to_csv(merged_csv, index=False)
    print(f"Merged traffic and weather data written to {merged_csv}")
