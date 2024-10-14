import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv('House_Price_Prediction.csv')  # Update the dataset filename if necessary

# Check if 'prefarea' and 'furnishingstatus' columns exist and encode them
if 'prefarea' in data.columns:
    data['prefarea'] = data['prefarea'].astype('category').cat.codes  # Encoding 'prefarea'
else:
    print("Warning: 'prefarea' column is missing from the dataset.")

if 'furnishingstatus' in data.columns:
    data['furnishingstatus'] = data['furnishingstatus'].astype('category').cat.codes  # Encoding 'furnishingstatus'
else:
    print("Warning: 'furnishingstatus' column is missing from the dataset.")

# Prepare feature and target variables
X = data[['bedrooms', 'bathrooms', 'area', 'prefarea']]  # Adjust features as needed
y = data['price']  # Target variable for house prices

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'house_price_model.pkl')

print("Model saved as 'house_price_model.pkl'")
