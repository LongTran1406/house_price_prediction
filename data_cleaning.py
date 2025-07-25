import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import requests
import json
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

key1 = os.getenv("LOCATIONIQ_KEY_1")
key2 = os.getenv("LOCATIONIQ_KEY_2")
key3 = os.getenv("LOCATIONIQ_KEY_3")
# Load data
df = pd.read_csv("raw_dataset.csv")
df = df.drop(columns=['id'])

# Visualize distributions
for col in ['bedroom_nums', 'bathroom_nums', 'car_spaces']:
    df[col].hist()
    plt.xlabel(col)
    plt.ylabel('Frequency')
    print(df[col].value_counts())
    plt.show()

# Fill NA with mode for categorical features

df = df.reset_index(drop=True)

# Geocoding setup
api_keys = [
    key1, 
    key2, 
    key3
]
url = "https://us1.locationiq.com/v1/search"
checkpoint_file = "geocode_partial.csv"

# Load checkpoint if exists
if os.path.exists(checkpoint_file):
    df_geo = pd.read_csv(checkpoint_file)
    done_indices = set(df_geo['index'])
else:
    done_indices = set()

total = len(df)
processed_count = 0

# Split indices into 3 roughly equal parts, one per API key
indices_chunks = []
chunk_size = (total + 2) // 3  # ceiling division

for i in range(3):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total)
    indices_chunks.append(list(range(start_idx, end_idx)))

def geocode_worker(indices, api_key):
    global processed_count
    results = []
    for i in indices:
        if i in done_indices:
            processed_count += 1
            print(f"Skipping {i}/{total} (already done). Progress: {processed_count}/{total}")
            continue

        params = {
            'key': api_key,
            'q': df['address'].iloc[i],
            'format': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                match = re.search(r'\b\d{4}\b', data[0]['display_name'])
                postcode = int(match.group()) if match else None
            else:
                lat, lon, postcode = None, None, None
        except Exception as e:
            print(f"Error on row {i}: {e}")
            lat, lon, postcode = None, None, None

        processed_count += 1
        print(f"Processed {i}/{total}. Progress: {processed_count}/{total}")

        # Append result immediately to checkpoint file
        pd.DataFrame([[i, lat, lon, postcode]], columns=['index', 'lat', 'lon', 'postcode']).to_csv(
            checkpoint_file, mode='a', index=False, header=not os.path.exists(checkpoint_file)
        )

        # Be polite to API
        time.sleep(0.7)
    return True

# Run 3 workers concurrently, each with its own API key
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for idx_chunk, api_key in zip(indices_chunks, api_keys):
        futures.append(executor.submit(geocode_worker, idx_chunk, api_key))
    for future in as_completed(futures):
        future.result()  # wait for completion

# After concurrency, load checkpoint & merge

