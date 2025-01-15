import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and explore the data

df = pd.read_csv("train.csv")

print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Step 2: Prepare features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

X = df[features]
y = df[target]

# Combine full and half baths
X['TotalBaths'] = X['FullBath'] + 0.5 * X['HalfBath']
X = X.drop(['FullBath', 'HalfBath'], axis=1)

print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully.")

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

# Step 6: Print model coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients)
print("\nIntercept:", model.intercept_)

# Step 7: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nVisualization complete. Check 'actual_vs_predicted.png' and 'feature_importance.png' for the plots.")

# Step 8: Example predictions
example_houses = pd.DataFrame({
    'GrLivArea': [1500, 2000, 2500],
    'BedroomAbvGr': [3, 4, 5],
    'TotalBaths': [2, 2.5, 3]
})

example_predictions = model.predict(example_houses)

print("\nExample Predictions:")
for i, (_, house) in enumerate(example_houses.iterrows()):
    print(f"House {i+1}:")
    print(f"  Area: {house['GrLivArea']} sq ft")
    print(f"  Bedrooms: {house['BedroomAbvGr']}")
    print(f"  Bathrooms: {house['TotalBaths']}")
    print(f"  Predicted Price: ${example_predictions[i]:,.2f}")
    print()