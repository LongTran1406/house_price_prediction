import pandas as pd
import re
# Load the file and drop bad header rows
df_geo = pd.read_csv("geocode_partial.csv")

# Drop rows where column names appeared again
df_geo = df_geo[df_geo['index'] != 'index']

# Convert columns back to correct dtypes
df_geo['index'] = df_geo['index'].astype(int)
df_geo = df_geo.sort_values(by='index').reset_index(drop=True)


# Reset index if needed
df_geo = df_geo.reset_index(drop=True)


df = pd.read_csv("raw_dataset.csv")

df = df.join(df_geo[['lat', 'lon', 'postcode']])


df['bedroom_nums'].fillna(df['bedroom_nums'].mode()[0], inplace=True)
df['bathroom_nums'].fillna(df['bathroom_nums'].mode()[0], inplace=True)
df['car_spaces'].fillna(df['car_spaces'].mode()[0], inplace=True)

# Convert land size to square meters
df['land_size_m2'] = df['land_size'].apply(
    lambda x: float(re.findall(r"\d+\.?\d*", x.replace(',', ''))[0]) * 10000 if isinstance(x, str) and 'ha' in x 
    else float(re.findall(r"\d+\.?\d*", x.replace(',', ''))[0]) if isinstance(x, str) and 'm' in x
    else None
)

df['land_size_m2'].fillna(df['land_size_m2'].mean(), inplace=True)
df.dropna(subset=['land_size_m2'], inplace=True)
df.drop(columns=['land_size'], inplace=True)

# Clean address and price
df['address'] = df['address'].apply(lambda x: x.replace('Address available on request', '') if isinstance(x, str) else x)
df['price'] = df['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
df['price_per_m2'] = df['price'] / df['land_size_m2']

df['lat'] = df['lat'].astype(float)
df['lon'] = df['lon'].astype(float)
df['postcode'] = df['postcode'].astype(float)

print(type(df.iloc[0]['lat']), type(df.iloc[0]['lat']), type(df.iloc[0]['postcode']))

city_coords = {
    'Sydney': (-33.8688, 151.2093),
    'Newcastle': (-32.9267, 151.7789),
    'Wollongong': (-34.4278, 150.8931)
}

def distance(lat, lon):
    dist = {city: (lat - lat_c)**2 + (lon - lon_c)**2 for city, (lat_c, lon_c) in city_coords.items()}
    nearest = min(dist, key=dist.get)
    return nearest, dist[nearest]

df['nearest_city'] = df.apply(lambda row: distance(row['lat'], row['lon'])[0], axis=1)
df['distance_to_nearest_city'] = df.apply(lambda row: distance(row['lat'], row['lon'])[1] * 10000, axis=1)

# Postcode price stats
postcode_avg_price = df.groupby('postcode')['price'].mean()
df['avg_price_by_postcode'] = df['postcode'].map(postcode_avg_price)

postcode_avg_price_per_m2 = df.groupby('postcode')['price_per_m2'].mean()
df['postcode_avg_price_per_m2'] = df['postcode'].map(postcode_avg_price_per_m2)

# Final output
df.to_csv("dataset_cleaned.csv", index=False)