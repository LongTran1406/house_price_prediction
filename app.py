import numpy as np
import flask
import pickle
from flask import Flask, jsonify, render_template, request
import requests
import re
import pandas as pd


app = Flask(__name__)

def load_model():
    with open('house_pricing_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_feature_engineering():
    with open('preprocessing.pkl', 'rb') as f:
        preprocess = pickle.load(f)
    return preprocess

preprocess = load_feature_engineering()
poly = preprocess['poly']
scaler = preprocess['scaler']
numeric_features = preprocess['numeric_features']
poly_feature_names = preprocess['poly_feature_names']
non_numeric_columns = preprocess['non_numeric_columns']


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    user_input = request.get_json()
    print(user_input)
    data = [[]]
    idx = 0
    for key, value in user_input.items():
        if idx == 0:
            data[0].append(value)
            idx += 1
        else:
            data[0].append(float(value))
    # data = np.array(data)
    print(data)
    address = data[0][0]
    bed_num = data[0][1]
    bath_num = data[0][2]
    parking = data[0][3]
    area = data[0][4]
    print(address)
    params = {
            'key': 'pk.d92da29e6c7492a1b8aede6a62a760b3',
            'q': address,
            'format': 'json'
        }
    url = "https://us1.locationiq.com/v1/search"
    response = requests.get(url, params=params, timeout=10).json()
    if isinstance(response, list) and len(response) > 0:
            lat = float(response[0]['lat'])
            lon = float(response[0]['lon'])
            match = re.search(r'\b\d{4}\b', response[0]['display_name'])
            postcode = float(match.group()) if match else None
    print(lat, lon, postcode)
    data[0].extend([lat, lon, postcode])
    city_coords = {
        'Sydney': (-33.8688, 151.2093),
        'Newcastle': (-32.9267, 151.7789),
        'Wollongong': (-34.4278, 150.8931)
    }

    def distance(lat, lon):
        dist = {city: (lat - lat_c)**2 + (lon - lon_c)**2 for city, (lat_c, lon_c) in city_coords.items()}
        nearest = min(dist, key=dist.get)
        return nearest, dist[nearest]

    nearest_city = distance(lat, lon)[0]
    distance_to_nearest_city = distance(lat, lon)[1] * 10000
    avg_price_by_postcode = df[df['postcode'] == postcode]['avg_price_by_postcode'].values[0]
    postcode_avg_price_per_m2 = df[df['postcode'] == postcode]['postcode_avg_price_per_m2'].values[0]
    print(nearest_city, distance_to_nearest_city, avg_price_by_postcode, postcode_avg_price_per_m2)
    building_density = (bed_num + bath_num + parking)/area * 100
    nearest_city_Newcastle = nearest_city_Sydney = nearest_city_Wollongong = False
    if nearest_city == 'Sydney':
        nearest_city_Sydney = True
    elif nearest_city == 'Wollongong': 
        nearest_city_Wollongong = True
    else:
        nearest_city_Newcastle = True
    new_data = [[bed_num, bath_num, parking, lat, lon, area, distance_to_nearest_city, avg_price_by_postcode
                ,postcode_avg_price_per_m2, building_density, nearest_city_Newcastle, nearest_city_Sydney, nearest_city_Wollongong]]
    # answer = model.predict(data)
    # print(answer)
    column_names = [
        'bedroom_nums',
        'bathroom_nums',
        'car_spaces',
        'lat',
        'lon',
        'land_size_m2',
        'distance_to_nearest_city',
        'avg_price_by_postcode',
        'postcode_avg_price_per_m2',
        'building_density',
        'nearest_city_Newcastle',
        'nearest_city_Sydney',
        'nearest_city_Wollongong'
    ]

    raw_df = pd.DataFrame(new_data, columns=column_names)
    X_poly = poly.transform(raw_df[numeric_features])
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
    X_enhanced = pd.concat([raw_df[non_numeric_columns], X_poly_df], axis=1)
    X_scaled = scaler.transform(X_enhanced)
    log_price_pred = model.predict(X_scaled)
    price_pred = np.expm1(log_price_pred)
    print(price_pred)
    return jsonify({'predicted_price': round(float(price_pred[0]), 2)})


model = load_model()
df = pd.read_csv("dataset_cleaned.csv")

if __name__ == '__main__':
    app.run(debug=True)
    