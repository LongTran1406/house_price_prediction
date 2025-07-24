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

from sklearn.model_selection import train_test_split
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr_pipeline = Pipeline([
    ('scaler', scaler),
    ('lr', lr)
])

lr_pipeline.fit(X_train_enhanced, y_train)
print("Train R2:", lr_pipeline.score(X_train_enhanced, y_train))
print("Test R2:", lr_pipeline.score(X_test_enhanced, y_test))


import xgboost as xgb

param_grid = {
    'n_estimators': [160, 200, 250],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
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

grid_search.fit(X_train_enhanced, y_train_log)


print("Best Parameters:", grid_search.best_params_)
print("Best Score (Neg MSE):", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_enhanced)
print("Test RMSE:", mean_squared_error(y_test_log, y_pred, squared=False))

import pickle

with open('house_pricing_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump({
        'poly': poly,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'poly_feature_names': poly_feature_names,
        'non_numeric_columns': X_train.drop(columns=numeric_features).columns.tolist()
    }, f)

