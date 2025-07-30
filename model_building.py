import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

scaler = StandardScaler()

df = pd.read_csv('dataset_cleaned.csv')

import sklearn
print(sklearn.__version__)


df = df.drop(columns = ['id', 'address'])
df['building_density'] = (df['bedroom_nums'] + df['bathroom_nums'] + df['car_spaces']) / df['land_size_m2'] * 100
df_dum = pd.get_dummies(df)

for col in df.columns:
    print(col)
    print(df[col].describe())
    print('---------------------')

X, y = df_dum.drop(columns = ['price_per_m2', 'price', 'postcode']), df_dum['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)


numeric_features = ['avg_price_by_postcode', 'bedroom_nums', 'bathroom_nums', 'car_spaces', 'lat', 'lon', 'land_size_m2', 'distance_to_nearest_city', 'postcode_avg_price_per_m2', 'building_density']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_poly = poly.fit_transform(X_train[numeric_features])

# Convert back to DataFrame with feature names
poly_feature_names = poly.get_feature_names_out(numeric_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_train.index)

# Combine with other features (e.g., dummies or other numeric features)
X_train_enhanced = pd.concat([X_train.drop(columns=numeric_features), X_poly_df], axis=1)

# Do same transformation on X_test before predicting
X_test_poly = poly.transform(X_test[numeric_features])
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)
X_test_enhanced = pd.concat([X_test.drop(columns=numeric_features), X_test_poly_df], axis=1)


lr = LinearRegression()

import mlflow


mlflow.set_tracking_uri("http://13.54.36.28:5000/")  # your actual EC2 IP
experiment_name = "new_experiment_name"
mlflow.set_experiment(experiment_name)
with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.runName", "LinearRegression_Baseline_TrainTestSplit")
    mlflow.set_tag("experiment_type", "baseline")
    mlflow.set_tag("model_type", "LinearRegression")
    
    lr_pipeline = Pipeline([
        ('scaler', scaler),
        ('lr', lr)
    ])
    
    lr_pipeline.fit(X_train_enhanced, y_train)
    print("Train R2:", lr_pipeline.score(X_train_enhanced, y_train))
    print("Test R2:", lr_pipeline.score(X_test_enhanced, y_test))


    mlflow.log_metric("train_r2", lr_pipeline.score(X_train_enhanced, y_train))
    mlflow.log_metric("test_r2", lr_pipeline.score(X_test_enhanced, y_test))
    mlflow.sklearn.log_model(lr_pipeline, "linear_model")
    

import xgboost as xgb

param_grid = {
    'n_estimators': [160, 200, 250],
    # 'max_depth': [4, 6, 8],
    # 'learning_rate': [0.01, 0.05, 0.1],
}

xgb_model = xgb.XGBRegressor(
    subsample=0.8,         # randomly sample rows for each tree
    colsample_bytree=0.8,  # randomly sample columns for each tree
    reg_alpha=15,           # L1 regularization
    reg_lambda=35,          # L2 regularization
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  
    cv=5,
    verbose=1,
    n_jobs=-1
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

#with mlflow.start_run():
 #   grid_search.fit(X_train_enhanced, y_train_log)
#
 #   # Log only best params and score
  #  mlflow.log_params(grid_search.best_params_)
   # mlflow.log_metric("best_cv_neg_mse", grid_search.best_score_)

#    best_model = grid_search.best_estimator_
 #   mlflow.sklearn.log_model(best_model, "best_model")

  #  preds = best_model.predict(X_test_enhanced)
   # rmse = np.sqrt(mean_squared_error(y_test_log, preds))
   # mlflow.log_metric("test_rmse", rmse)


import pickle

#with open('house_pricing_model.pkl', 'wb') as f:
 #   pickle.dump(best_model, f)

with open('linear_model.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump({
        'poly': poly,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'poly_feature_names': poly_feature_names,
        'non_numeric_columns': X_train.drop(columns=numeric_features).columns.tolist()
    }, f)

import pandas as pd

# Build test row (based on your data)
test_row = pd.DataFrame([{
    'bedroom_nums': 4.0,
    'bathroom_nums': 2,
    'car_spaces': 2.0,
    'lat': -33.8921,
    'lon': 150.8844,
    'land_size_m2': 600.0,
    'distance_to_nearest_city': 1063.635874,
    'avg_price_by_postcode': 1680905.273,
    'postcode_avg_price_per_m2': 2535.471305,
    'building_density': 0.86,
    'nearest_city_Newcastle': False,
    'nearest_city_Sydney': True,
    'nearest_city_Wollongong': False
}])


test_poly = poly.transform(test_row[numeric_features])
test_poly_df = pd.DataFrame(test_poly, columns=poly.get_feature_names_out(numeric_features))
test_enhanced = pd.concat(
    [test_row.drop(columns=numeric_features).reset_index(drop=True), test_poly_df],
    axis=1
)
print(lr_pipeline.predict(test_enhanced))