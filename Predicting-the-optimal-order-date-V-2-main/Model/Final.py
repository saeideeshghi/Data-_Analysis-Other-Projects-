#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow


# # Data Analysis and Visualization

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the 'img' and 'csv' directories exist
os.makedirs('img', exist_ok=True)
os.makedirs('csv', exist_ok=True)

# Load the dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Ensure the 'OrderDate' column is converted to datetime type
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

# Check for invalid values in 'OrderDate'
if df['OrderDate'].isnull().any():
    print("Some values in 'OrderDate' could not be converted to datetime.")

# Data analysis and visualization settings
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
plt.savefig(os.path.join('img', 'stock_optimal_order_relationship.png'))
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
plt.savefig(os.path.join('img', 'average_optimal_order_time.png'))
plt.show()
plt.close()

# 3. Impact of Seasonality on Optimal Order Time
plt.figure(figsize=(10, 6))
sns.boxplot(x='Seasonality', y='OptimalOrderTime', data=df)
plt.title('Impact of Seasonality on Optimal Order Time')
plt.xlabel('Season')
plt.ylabel('Optimal Order Time (Days)')
plt.tight_layout()
plt.savefig(os.path.join('img', 'seasonality_impact_on_order_time.png'))
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
plt.savefig(os.path.join('img', 'optimal_order_date_distribution.png'))
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
plt.savefig(os.path.join('img', 'average_optimal_order_time_by_day.png'))
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
plt.savefig(os.path.join('img', 'price_optimal_order_relationship.png'))
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
plt.savefig(os.path.join('img', 'promotion_impact_on_sales.png'))
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
plt.savefig(os.path.join('img', 'sales_trend_over_time.png'))
plt.show()
plt.close()

# 9. Histogram of Stock Cover Days
plt.figure(figsize=(14, 6))
sns.histplot(df['StockCoverDays'], kde=True, bins=30)
plt.title('Distribution of Stock Cover Days')
plt.xlabel('Stock Cover Days')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join('img', 'stock_cover_days_distribution.png'))
plt.show()
plt.close()

# 10. Correlation Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join('img', 'correlation_heatmap.png'))
plt.show()
plt.close()

# 11. Boxplot of Weekly Sales by Seasonality
plt.figure(figsize=(14, 8))
sns.boxplot(x='Seasonality', y='WeeklySales', data=df)
plt.title('Weekly Sales by Seasonality')
plt.xlabel('Seasonality')
plt.ylabel('Weekly Sales')
plt.tight_layout()
plt.savefig(os.path.join('img', 'weekly_sales_by_seasonality.png'))
plt.show()
plt.close()

# 12. Violin Plot of Optimal Order Time by Product
plt.figure(figsize=(14, 8))
sns.violinplot(x='Product', y='OptimalOrderTime', data=df)
plt.title('Optimal Order Time by Product')
plt.xlabel('Product')
plt.ylabel('Optimal Order Time (Days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join('img', 'optimal_order_time_violin.png'))
plt.show()
plt.close()

# 13. Line Plot of Inventory Turnover Ratio over Time
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
plt.savefig(os.path.join('img', 'inventory_turnover_trend.png'))
plt.show()
plt.close()

print("All plots have been successfully generated and saved in the 'img' folder.")


# # Machine Learning Models and Evaluation

# In[2]:


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

# Ensure the 'img' and 'csv' directories exist
os.makedirs('img', exist_ok=True)
os.makedirs('csv', exist_ok=True)

# Load the dataset
df = pd.read_csv('extended_inventory_dataset_35k.csv')

# Add 'OrderID' column if it does not exist
if 'OrderID' not in df.columns:
    df['OrderID'] = df.index + 1

# Convert 'OrderDate', 'OptimalOrderDate', and 'ExpirationDate' columns to datetime type
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

# 1. Reduced Complexity for RandomForest
rf_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=4)
rf_model.fit(X_train_scaled, y_train)
y_rf_pred = rf_model.predict(X_test_scaled)

# 2. Reduced Complexity for XGBoost
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

# 6. Enhanced Neural Network
nn_model = Sequential()
nn_model.add(Input(shape=(X_train_scaled.shape[1],)))
nn_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Neural Network model
history = nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)
y_nn_pred = nn_model.predict(X_test_scaled).flatten()

# Evaluate the models
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Calculate results for each model
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

print(results)

# Save the results to an Excel file in the 'csv' folder
results.to_excel(os.path.join('csv', 'final_model_results.xlsx'), index=False)

# Plot comparison charts
models = ['Random Forest', 'XGBoost', 'Linear Regression',
          'Lasso', 'Ridge', 'Neural Network']
predictions = [y_rf_pred, y_xgb_pred, y_lr_pred,
               y_lasso_pred, y_ridge_pred, y_nn_pred]

# 1. Actual vs Predicted, Residual Plot, Histogram, and Q-Q Plot for each model
for i, model in enumerate(models):
    # Actual vs Predicted
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=y_test, y=predictions[i], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Days Until Optimal Order')
    plt.ylabel(f'Predicted Days Until Optimal Order ({model})')
    plt.title(f'Actual vs Predicted Days Until Optimal Order ({model})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('img', f'actual_vs_predicted_{model.replace(" ", "_").lower()}.png'))
    plt.show()

    # Residual Plot
    residuals = predictions[i] - y_test
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=predictions[i], y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(f'Predicted Days Until Optimal Order ({model})')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({model})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('img', f'residual_plot_{model.replace(" ", "_").lower()}.png'))
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(14, 8))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f'Distribution of Residuals ({model})')
    plt.xlabel('Residuals (Days)')
    plt.ylabel('Frequency')
    plt.axvline(residuals.mean(), color='r', linestyle='--',
                label=f'Mean Residual: {residuals.mean():.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('img', f'histogram_residuals_{model.replace(" ", "_").lower()}.png'))
    plt.show()

    # Q-Q Plot of Residuals
    plt.figure(figsize=(14, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Residuals ({model})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('img', f'qq_plot_residuals_{model.replace(" ", "_").lower()}.png'))
    plt.show()

# 2. Feature Importance for RandomForest
plt.figure(figsize=(14, 10))
importances_rf = rf_model.feature_importances_
importance_df_rf = pd.DataFrame(
    {'Feature': X.columns, 'Importance': importances_rf})
importance_df_rf = importance_df_rf.sort_values(
    by='Importance', ascending=False).head(15)  # Display top 15 features
sns.barplot(x='Importance', y='Feature',
            data=importance_df_rf, palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('img', 'feature_importances_random_forest.png'))
plt.show()

# 3. Feature Importance for XGBoost
plt.figure(figsize=(14, 10))
importances_xgb = xgb_model.feature_importances_
importance_df_xgb = pd.DataFrame(
    {'Feature': X.columns, 'Importance': importances_xgb})
importance_df_xgb = importance_df_xgb.sort_values(
    by='Importance', ascending=False).head(15)  # Display top 15 features
sns.barplot(x='Importance', y='Feature',
            data=importance_df_xgb, palette='magma')
plt.title('Top 15 Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('img', 'feature_importances_xgboost.png'))
plt.show()

# 4. Neural Network Training Loss
plt.figure(figsize=(14, 8))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Neural Network Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('img', 'neural_network_training_loss.png'))
plt.show()

# 5. Box Plot of Errors for Each Model
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
plt.xlabel('Model')
plt.ylabel('Prediction Error (Days)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('img', 'box_plot_prediction_errors.png'))
plt.show()

# 6. Comparison of Evaluation Metrics
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


# In[ ]:




