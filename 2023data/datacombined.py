import os
import glob
import pandas as pd
import datetime
from meteostat import Point, Daily

def combine_site_100_data(root_dir, output_csv):
    """
    Scans each monthly subfolder (VSDATA_2023MM),
    looks for CSV files, filters NB_SCATS_SITE == 100,
    and writes a single combined CSV.
    """
    monthly_dirs = glob.glob(os.path.join(root_dir, "VSDATA_2023*"))
    monthly_dirs.sort()

    filtered_dfs = []

    for month_dir in monthly_dirs:
        csv_files = glob.glob(os.path.join(month_dir, "*.csv"))
        csv_files.sort()

        if not csv_files:
            print(f"No CSV files found in {month_dir}. Skipping.")
            continue

        print(f"\nProcessing folder: {os.path.basename(month_dir)}")
        
        for csv_file in csv_files:
            print(f"  Reading {csv_file}")
            df = pd.read_csv(csv_file)

            # Filter for site 100
            df_site100 = df[df["NB_SCATS_SITE"] == 100]
            if not df_site100.empty:
                filtered_dfs.append(df_site100)

    if not filtered_dfs:
        print("No rows found for NB_SCATS_SITE == 100 in any subfolder.")
        return None

    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(combined_df)} rows to {output_csv}")
    
    return combined_df


if __name__ == "__main__":
    # 1) Combine site-100 data into one CSV
    root_path = "/Users/shrutichintalapati/Documents/nn project/2023data"
    traffic_output_csv = "/Users/shrutichintalapati/Documents/nn project/site100_all.csv"
    
    combined_df = combine_site_100_data(root_path, traffic_output_csv)
    if combined_df is None:
        # If no rows found, stop here
        print("No site 100 data was created.")
    else:
        # 2) Prepare the combined data for merging with weather
        #    We assume 'QT_INTERVAL_COUNT' holds a daily date/time string like '2023-01-01 00:00:00...'
        #    Convert it to an actual date column named 'Date'
        combined_df['Date'] = pd.to_datetime(combined_df['QT_INTERVAL_COUNT']).dt.date
        
        # 3) Fetch daily weather data for Victoria using Meteostat
        start = datetime.datetime(2023, 1, 1)
        end   = datetime.datetime(2023, 12, 31)
        victoria_loc = Point(-37.8136, 144.9631)  # Melbourne lat/lon
        
        weather_data = Daily(victoria_loc, start, end).fetch()
