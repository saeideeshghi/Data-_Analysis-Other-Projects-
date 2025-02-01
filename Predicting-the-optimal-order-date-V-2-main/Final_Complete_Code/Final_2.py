#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow joblib ipywidgets scipy


# # Data Analysis and Visualization

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the 'img/dataset' and 'csv' directories exist
os.makedirs(os.path.join('img', 'dataset'), exist_ok=True)
os.makedirs('csv', exist_ok=True)

# Load the dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Convert 'OrderDate' to datetime type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

# Check for invalid 'OrderDate' values
if df['OrderDate'].isnull().any():
    print("Some values in 'OrderDate' could not be converted to datetime.")

# Data visualization settings
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# 1. Relationship between Stock on Hand and Optimal Order Time
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='StockOnHand', y='OptimalOrderTime', hue='Product', alpha=0.6)
plt.title('Relationship between Stock on Hand and Optimal Order Time')
plt.xlabel('Stock on Hand')
plt.ylabel('Optimal Order Time (Days)')
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'stock_optimal_order_relationship.png'))
plt.show()
plt.close()

# 2. Average Optimal Order Time by Product
avg_order_time = df.groupby('Product')['OptimalOrderTime'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
avg_order_time.plot(kind='bar')
plt.title('Average Optimal Order Time by Product')
plt.xlabel('Product')
plt.ylabel('Average Optimal Order Time (Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'average_optimal_order_time.png'))
plt.show()
plt.close()

# 3. Impact of Seasonality on Optimal Order Time
plt.figure(figsize=(10, 6))
sns.boxplot(x='Seasonality', y='OptimalOrderTime', data=df)
plt.title('Impact of Seasonality on Optimal Order Time')
plt.xlabel('Season')
plt.ylabel('Optimal Order Time (Days)')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'seasonality_impact_on_order_time.png'))
plt.show()
plt.close()

# 4. Distribution of Optimal Order Dates
plt.figure(figsize=(14, 6))
df['OptimalOrderDate'] = pd.to_datetime(df['OptimalOrderDate'], errors='coerce')
df['OptimalOrderDate'].dropna().dt.to_pydatetime()
df['OptimalOrderDate'].hist(bins=50)
plt.title('Distribution of Optimal Order Dates')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'optimal_order_date_distribution.png'))
plt.show()
plt.close()

# 5. Average Optimal Order Time by Day of Week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_order_time_day = df.groupby('DayOfWeek')['OptimalOrderTime'].mean()
plt.figure(figsize=(10, 6))
avg_order_time_day.plot(kind='bar')
plt.title('Average Optimal Order Time by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Optimal Order Time (Days)')
plt.xticks(range(7), days, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'average_optimal_order_time_by_day.png'))
plt.show()
plt.close()

# 6. Relationship between Price and Optimal Order Time
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Price', y='OptimalOrderTime', hue='Product')
plt.title('Relationship between Price and Optimal Order Time')
plt.xlabel('Price')
plt.ylabel('Optimal Order Time (Days)')
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'price_optimal_order_relationship.png'))
plt.show()
plt.close()

# 7. Impact of Promotions on Weekly Sales
plt.figure(figsize=(10, 6))
sns.boxplot(x='PromotionActive', y='WeeklySales', data=df)
plt.title('Impact of Promotions on Weekly Sales')
plt.xlabel('Promotion Active')
plt.ylabel('Weekly Sales')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'promotion_impact_on_sales.png'))
plt.show()
plt.close()

# 8. Weekly Sales Trend Over Time
df['YearMonth'] = df['OrderDate'].dt.to_period('M')
monthly_sales = df.groupby('YearMonth')['WeeklySales'].mean().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='WeeklySales')
plt.title('Weekly Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'sales_trend_over_time.png'))
plt.show()
plt.close()

# 9. Histogram of Stock Cover Days
plt.figure(figsize=(14, 6))
sns.histplot(df['StockCoverDays'], kde=True, bins=30)
plt.title('Distribution of Stock Cover Days')
plt.xlabel('Stock Cover Days')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'stock_cover_days_distribution.png'))
plt.show()
plt.close()

# 10. Correlation Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'correlation_heatmap.png'))
plt.show()
plt.close()

# 11. Boxplot of Weekly Sales by Seasonality
plt.figure(figsize=(14, 8))
sns.boxplot(x='Seasonality', y='WeeklySales', data=df)
plt.title('Weekly Sales by Seasonality')
plt.xlabel('Seasonality')
plt.ylabel('Weekly Sales')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'weekly_sales_by_seasonality.png'))
plt.show()
plt.close()

# 12. Violin plot of Optimal Order Time by Product
plt.figure(figsize=(14, 8))
sns.violinplot(x='Product', y='OptimalOrderTime', data=df)
plt.title('Optimal Order Time by Product')
plt.xlabel('Product')
plt.ylabel('Optimal Order Time (Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'optimal_order_time_violin.png'))
plt.show()
plt.close()

# 13. Line plot of Inventory Turnover Ratio over Time
df['YearMonth'] = df['OrderDate'].dt.to_period('M')
monthly_turnover = df.groupby('YearMonth')['InventoryTurnoverRatio'].mean().reset_index()
monthly_turnover['YearMonth'] = monthly_turnover['YearMonth'].astype(str)
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_turnover, x='YearMonth', y='InventoryTurnoverRatio')
plt.title('Inventory Turnover Ratio over Time')
plt.xlabel('Date')
plt.ylabel('Average Inventory Turnover Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join('img', 'dataset', 'inventory_turnover_trend.png'))
plt.show()
plt.close()

print("All plots have been successfully generated and saved in the 'img/dataset' folder.")


# # Model Training, Evaluation, and Visualization

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats  # Importing scipy library

# Ensure the 'img/model_output' and 'csv' directories exist
os.makedirs(os.path.join('img', 'model_output'), exist_ok=True)
os.makedirs('csv', exist_ok=True)

# Load the dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Add 'OrderID' column if it does not exist
if 'OrderID' not in df.columns:
    df['OrderID'] = df.index + 1

# Convert date columns to datetime type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df['OptimalOrderDate'] = pd.to_datetime(df['OptimalOrderDate'], errors='coerce')
df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')

# Calculate the number of days until the optimal order date (target variable)
df['DaysUntilOptimalOrder'] = (df['OptimalOrderDate'] - df['OrderDate']).dt.days

# Remove NaN or negative values in 'DaysUntilOptimalOrder'
df = df.dropna(subset=['DaysUntilOptimalOrder'])
df = df[df['DaysUntilOptimalOrder'] > 0]

# Define features and target variable
features = ['LeadTime', 'StockOnHand', 'CapitalRecord', 'WeeklySales',
            'OrderToReceiveTime', 'HoldingCost', 'DaysUntilExpiration']
X = pd.get_dummies(df[features], drop_first=True)
y = df['DaysUntilOptimalOrder']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
# 1. RandomForest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=4)
rf_model.fit(X_train_scaled, y_train)
y_rf_pred = rf_model.predict(X_test_scaled)

# 2. XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_xgb_pred = xgb_model.predict(X_test_scaled)

# 3. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_lr_pred = lr_model.predict(X_test_scaled)

# 4. Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
y_lasso_pred = lasso_model.predict(X_test_scaled)

# 5. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_ridge_pred = ridge_model.predict(X_test_scaled)

# 6. Neural Network Model
nn_model = Sequential()
nn_model.add(Input(shape=(X_train_scaled.shape[1],)))
nn_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model
history = nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)
y_nn_pred = nn_model.predict(X_test_scaled).flatten()

# Model evaluation function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Compute evaluation results for each model
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Linear Regression',
              'Lasso', 'Ridge', 'Neural Network'],
    'MAE': [
        evaluate_model(y_test, y_rf_pred)[0],
        evaluate_model(y_test, y_xgb_pred)[0],
        evaluate_model(y_test, y_lr_pred)[0],
        evaluate_model(y_test, y_lasso_pred)[0],
        evaluate_model(y_test, y_ridge_pred)[0],
        evaluate_model(y_test, y_nn_pred)[0]
    ],
    'MSE': [
        evaluate_model(y_test, y_rf_pred)[1],
        evaluate_model(y_test, y_xgb_pred)[1],
        evaluate_model(y_test, y_lr_pred)[1],
        evaluate_model(y_test, y_lasso_pred)[1],
        evaluate_model(y_test, y_ridge_pred)[1],
        evaluate_model(y_test, y_nn_pred)[1]
    ],
    'R^2': [
        evaluate_model(y_test, y_rf_pred)[2],
        evaluate_model(y_test, y_xgb_pred)[2],
        evaluate_model(y_test, y_lr_pred)[2],
        evaluate_model(y_test, y_lasso_pred)[2],
        evaluate_model(y_test, y_ridge_pred)[2],
        evaluate_model(y_test, y_nn_pred)[2]
    ]
})

# Save the evaluation results to an Excel file
results.to_excel(os.path.join('csv', 'final_model_results.xlsx'), index=False)
print(results)

# Generate comparison plots
models = ['Random Forest', 'XGBoost', 'Linear Regression', 'Lasso', 'Ridge', 'Neural Network']
predictions = [y_rf_pred, y_xgb_pred, y_lr_pred, y_lasso_pred, y_ridge_pred, y_nn_pred]

# Plot Actual vs Predicted and Residuals for each model
for i, model in enumerate(models):
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=y_test, y=predictions[i], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Days Until Optimal Order')
    plt.ylabel(f'Predicted Days Until Optimal Order ({model})')
    plt.title(f'Actual vs Predicted Days Until Optimal Order ({model})')
    plt.tight_layout()
    plt.savefig(os.path.join('img', 'model_output', f'actual_vs_predicted_{model.lower().replace(" ", "_")}.png'))
    plt.show()

    # 2. Residual Plot
    residuals = predictions[i] - y_test
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=predictions[i], y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(f'Predicted Days Until Optimal Order ({model})')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({model})')
    plt.tight_layout()
    plt.savefig(os.path.join('img', 'model_output', f'residual_plot_{model.lower().replace(" ", "_")}.png'))
    plt.show()

    # 3. Histogram of Residuals
    plt.figure(figsize=(14, 8))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f'Distribution of Residuals ({model})')
    plt.xlabel('Residuals (Days)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join('img', 'model_output', f'residual_histogram_{model.lower().replace(" ", "_")}.png'))
    plt.show()

# Feature importance for Random Forest
plt.figure(figsize=(14, 10))
importances_rf = rf_model.feature_importances_
importance_df_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importances_rf}).sort_values(by='Importance', ascending=False).head(15)
sns.barplot(x='Importance', y='Feature', data=importance_df_rf, palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_output', 'rf_feature_importances.png'))
plt.show()

# Feature importance for XGBoost
plt.figure(figsize=(14, 10))
importances_xgb = xgb_model.feature_importances_
importance_df_xgb = pd.DataFrame({'Feature': X.columns, 'Importance': importances_xgb}).sort_values(by='Importance', ascending=False).head(15)
sns.barplot(x='Importance', y='Feature', data=importance_df_xgb, palette='magma')
plt.title('Top 15 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_output', 'xgb_feature_importances.png'))
plt.show()

# Neural Network Training Loss
plt.figure(figsize=(14, 8))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Neural Network Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_output', 'nn_training_loss.png'))
plt.show()

# Box Plot of Prediction Errors for Each Model
plt.figure(figsize=(14, 10))
errors = {
    'Random Forest': y_rf_pred - y_test,
    'XGBoost': y_xgb_pred - y_test,
    'Linear Regression': y_lr_pred - y_test,
    'Lasso': y_lasso_pred - y_test,
    'Ridge': y_ridge_pred - y_test,
    'Neural Network': y_nn_pred - y_test
}
sns.boxplot(data=pd.DataFrame(errors), palette='Set3')
plt.title('Box Plot of Prediction Errors for Each Model')
plt.tight_layout()
plt.savefig(os.path.join('img', 'model_output', 'boxplot_prediction_errors.png'))
plt.show()


# Comparison of Evaluation Metrics
metrics = ['MAE', 'MSE', 'R^2']
for metric in metrics:
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y=metric, data=results, palette='coolwarm')
    plt.title(f'Comparison of {metric} Across Models')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('img', f'comparison_{metric.lower()}.png'))
    plt.show()

# Ensure that only test rows are used for predictions
output_df = df.loc[X_test.index, ['OrderID', 'OrderDate', 'OptimalOrderDate']].copy()

# Add predictions from each model to the output dataset
output_df['PredictedDaysUntilOptimalOrder_RF'] = y_rf_pred
output_df['PredictedDaysUntilOptimalOrder_XGBoost'] = y_xgb_pred
output_df['PredictedDaysUntilOptimalOrder_LR'] = y_lr_pred
output_df['PredictedDaysUntilOptimalOrder_Lasso'] = y_lasso_pred
output_df['PredictedDaysUntilOptimalOrder_Ridge'] = y_ridge_pred
output_df['PredictedDaysUntilOptimalOrder_NN'] = y_nn_pred

# Calculate predicted optimal order dates for each model
output_df['PredictedOptimalOrderDate_RF'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_RF'], unit='D')
output_df['PredictedOptimalOrderDate_XGBoost'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_XGBoost'], unit='D')
output_df['PredictedOptimalOrderDate_LR'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_LR'], unit='D')
output_df['PredictedOptimalOrderDate_Lasso'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_Lasso'], unit='D')
output_df['PredictedOptimalOrderDate_Ridge'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_Ridge'], unit='D')
output_df['PredictedOptimalOrderDate_NN'] = output_df['OrderDate'] + pd.to_timedelta(output_df['PredictedDaysUntilOptimalOrder_NN'], unit='D')

# Select desired columns for the final output
output_final = output_df[['OrderID', 'OrderDate', 'OptimalOrderDate',
                          'PredictedOptimalOrderDate_RF', 'PredictedOptimalOrderDate_XGBoost',
                          'PredictedOptimalOrderDate_LR', 'PredictedOptimalOrderDate_Lasso',
                          'PredictedOptimalOrderDate_Ridge', 'PredictedOptimalOrderDate_NN']]

# Save the final output to an Excel file in the 'csv' folder
output_final.to_excel(os.path.join('csv', 'inventory_predictions_output_test_noisy_v3-1.xlsx'), index=False)
print("The output dataset with predictions has been saved to 'csv/inventory_predictions_output_test_noisy_v3-1.xlsx'.")


# # Model Training and Saving

# In[3]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import regularizers
import joblib

# Ensure necessary directories exist for saving models and datasets
os.makedirs('models', exist_ok=True)
os.makedirs('csv', exist_ok=True)

# Load dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Add 'OrderID' column if it doesn't exist
if 'OrderID' not in df.columns:
    df['OrderID'] = df.index + 1

# Convert date columns to datetime type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df['OptimalOrderDate'] = pd.to_datetime(df['OptimalOrderDate'], errors='coerce')
df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')

# Calculate the number of days until the optimal order date (target variable)
df['DaysUntilOptimalOrder'] = (df['OptimalOrderDate'] - df['OrderDate']).dt.days

# Remove NaN or negative values in 'DaysUntilOptimalOrder'
df = df.dropna(subset=['DaysUntilOptimalOrder'])
df = df[df['DaysUntilOptimalOrder'] > 0]

# Define features and target variable
features = ['LeadTime', 'StockOnHand', 'CapitalRecord', 'WeeklySales',
            'OrderToReceiveTime', 'HoldingCost', 'DaysUntilExpiration']
X = df[features]
y = df['DaysUntilOptimalOrder']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
# 1. RandomForest Model
rf_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=4)
rf_model.fit(X_train_scaled, y_train)

# 2. XGBoost Model
xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 3. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 4. Lasso Regression Model
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

# 5. Ridge Regression Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# 6. Neural Network Model
nn_model = Sequential()
nn_model.add(Input(shape=(X_train_scaled.shape[1],)))
nn_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model
nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)

# Save the models and scaler
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
joblib.dump(rf_model, os.path.join('models', 'rf_model.pkl'))
joblib.dump(xgb_model, os.path.join('models', 'xgb_model.pkl'))
joblib.dump(lr_model, os.path.join('models', 'lr_model.pkl'))
joblib.dump(lasso_model, os.path.join('models', 'lasso_model.pkl'))
joblib.dump(ridge_model, os.path.join('models', 'ridge_model.pkl'))
nn_model.save(os.path.join('models', 'nn_model.h5'))

print("Models and scaler have been successfully saved.")


# # Loading Models and Creating a User Interface

# In[1]:


# Ensure the 'models', 'img', and 'csv' directories exist
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import scipy.stats as stats  # Ensure scipy is imported for Q-Q plots

# Ensure inline plotting for Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the scaler and models
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
rf_model = joblib.load(os.path.join('models', 'rf_model.pkl'))
xgb_model = joblib.load(os.path.join('models', 'xgb_model.pkl'))
lr_model = joblib.load(os.path.join('models', 'lr_model.pkl'))
lasso_model = joblib.load(os.path.join('models', 'lasso_model.pkl'))
ridge_model = joblib.load(os.path.join('models', 'ridge_model.pkl'))
nn_model = load_model(os.path.join('models', 'nn_model.h5'), compile=False)  # Load model without compiling

# Load the dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Add 'OrderID' column if it does not exist
if 'OrderID' not in df.columns:
    df['OrderID'] = df.index + 1

# Convert date columns to datetime type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df['OptimalOrderDate'] = pd.to_datetime(df['OptimalOrderDate'], errors='coerce')
df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')

# Extract list of products (assuming 'Product' column exists)
if 'Product' in df.columns:
    product_list = df['Product'].unique()
else:
    print("Column 'Product' not found in the dataset.")
    raise ValueError("Column 'Product' not found in the dataset.")

# Calculate min and max for each numeric feature
numeric_features = ['LeadTime', 'StockOnHand', 'CapitalRecord', 'WeeklySales',
                    'OrderToReceiveTime', 'HoldingCost', 'DaysUntilExpiration']

feature_mins = df[numeric_features].min()
feature_maxs = df[numeric_features].max()

# Define a common style for all widgets to adjust description width
style = {'description_width': 'initial'}

# User Interface Widgets
product_dropdown = widgets.Dropdown(
    options=product_list,
    description='Product:',
    disabled=False,
    style=style
)

lead_time_slider = widgets.IntSlider(
    min=int(feature_mins['LeadTime']),
    max=int(feature_maxs['LeadTime']),
    description='Lead Time:',
    disabled=False,
    style=style
)

stock_on_hand_slider = widgets.IntSlider(
    min=int(feature_mins['StockOnHand']),
    max=int(feature_maxs['StockOnHand']),
    description='Stock On Hand:',
    disabled=False,
    style=style
)

capital_record_slider = widgets.FloatSlider(
    min=float(feature_mins['CapitalRecord']),
    max=float(feature_maxs['CapitalRecord']),
    description='Capital Record:',
    step=0.1,
    disabled=False,
    style=style
)

weekly_sales_slider = widgets.IntSlider(
    min=int(feature_mins['WeeklySales']),
    max=int(feature_maxs['WeeklySales']),
    description='Weekly Sales:',
    disabled=False,
    style=style
)

order_to_receive_time_slider = widgets.IntSlider(
    min=int(feature_mins['OrderToReceiveTime']),
    max=int(feature_maxs['OrderToReceiveTime']),
    description='Order to Receive Time:',
    disabled=False,
    style=style
)

holding_cost_slider = widgets.FloatSlider(
    min=float(feature_mins['HoldingCost']),
    max=float(feature_maxs['HoldingCost']),
    description='Holding Cost:',
    step=0.1,
    disabled=False,
    style=style
)

days_until_expiration_slider = widgets.IntSlider(
    min=int(feature_mins['DaysUntilExpiration']),
    max=int(feature_maxs['DaysUntilExpiration']),
    description='Days Until Expiration:',
    disabled=False,
    style=style
)

order_date_picker = widgets.DatePicker(
    description='Order Date:',
    disabled=False,
    style=style
)

predict_button = widgets.Button(
    description='Predict',
    button_style='success',
    tooltip='Predict Optimal Order Date',
    icon='check',
    style=style
)

output = widgets.Output()

# Define function to update field values based on selected product
def update_fields(change):
    selected_product = product_dropdown.value
    product_data = df[df['Product'] == selected_product].iloc[0]
    lead_time_slider.value = int(product_data['LeadTime'])
    stock_on_hand_slider.value = int(product_data['StockOnHand'])
    capital_record_slider.value = float(product_data['CapitalRecord'])
    weekly_sales_slider.value = int(product_data['WeeklySales'])
    order_to_receive_time_slider.value = int(product_data['OrderToReceiveTime'])
    holding_cost_slider.value = float(product_data['HoldingCost'])
    days_until_expiration_slider.value = int(product_data['DaysUntilExpiration'])
    order_date_picker.value = product_data['OrderDate'].date()

# Connect product selection to the update function
product_dropdown.observe(update_fields, names='value')

# Initialize fields with the first product's data
update_fields(None)

# Define the prediction function
def on_predict_button_clicked(b):
    with output:
        clear_output()
        # Collect input data
        input_data = pd.DataFrame({
            'LeadTime': [lead_time_slider.value],
            'StockOnHand': [stock_on_hand_slider.value],
            'CapitalRecord': [capital_record_slider.value],
            'WeeklySales': [weekly_sales_slider.value],
            'OrderToReceiveTime': [order_to_receive_time_slider.value],
            'HoldingCost': [holding_cost_slider.value],
            'DaysUntilExpiration': [days_until_expiration_slider.value]
        })
        
        order_date = order_date_picker.value
        
        # Preprocess input data
        X_input = scaler.transform(input_data)
        
        # Predict with all models
        predicted_days_rf = rf_model.predict(X_input)[0]
        predicted_days_xgb = xgb_model.predict(X_input)[0]
        predicted_days_lr = lr_model.predict(X_input)[0]
        predicted_days_lasso = lasso_model.predict(X_input)[0]
        predicted_days_ridge = ridge_model.predict(X_input)[0]
        predicted_days_nn = nn_model.predict(X_input)[0][0]
        
        # Calculate predicted optimal order dates
        predicted_date_rf = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_rf, unit='D')
        predicted_date_xgb = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_xgb, unit='D')
        predicted_date_lr = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_lr, unit='D')
        predicted_date_lasso = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_lasso, unit='D')
        predicted_date_ridge = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_ridge, unit='D')
        predicted_date_nn = pd.to_datetime(order_date) + pd.to_timedelta(predicted_days_nn, unit='D')
        
        # Create a DataFrame of results with English text
        results_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Linear Regression',
                      'Lasso Regression', 'Ridge Regression', 'Neural Network'],
            'Days Until Optimal Order Date': [predicted_days_rf, predicted_days_xgb,
                                              predicted_days_lr, predicted_days_lasso,
                                              predicted_days_ridge, predicted_days_nn],
            'Optimal Order Date': [predicted_date_rf.date(), predicted_date_xgb.date(),
                                   predicted_date_lr.date(), predicted_date_lasso.date(),
                                   predicted_date_ridge.date(), predicted_date_nn.date()]
        })
        
        # Display the results
        display(results_df)
        
        # Plot comparison chart 1
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Model', y='Days Until Optimal Order Date', data=results_df, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title('Comparison of Predicted Days Until Optimal Order by Models')
        ax.set_xlabel('Model')
        ax.set_ylabel('Days')
        plt.tight_layout()
        plt.savefig(os.path.join('img', 'comparison_predicted_days_models.png'))
        display(fig)  # Use display to show the plot
        plt.close(fig)
        
        # Plot comparison chart 2
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(x='Model', y='Days Until Optimal Order Date', data=results_df, palette='magma', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Comparison of Optimal Order Dates Predicted by Models')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Days')
        plt.tight_layout()
        plt.savefig(os.path.join('img', 'comparison_optimal_order_dates_models.png'))
        display(fig2)  # Use display to show the plot
        plt.close(fig2)

# Connect the predict button to the prediction function
predict_button.on_click(on_predict_button_clicked)

# Organize widgets into a grid layout
input_widgets = widgets.GridBox(
    children=[
        product_dropdown,
        order_date_picker,
        lead_time_slider,
        stock_on_hand_slider,
        capital_record_slider,
        weekly_sales_slider,
        order_to_receive_time_slider,
        holding_cost_slider,
        days_until_expiration_slider,
        predict_button
    ],
    layout=widgets.Layout(
        width='100%',
        grid_template_columns='repeat(2, 50%)',
        grid_template_rows='auto',
        grid_gap='10px 10px'
    )
)

# Display the widgets
display(widgets.VBox([
    input_widgets,
    output
]))


# In[ ]:




