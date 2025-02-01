#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pandas numpy scikit-learn xgboost tensorflow joblib matplotlib seaborn ipywidgets scipy


# # Training and Saving Models

# In[2]:


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

# Ensure the 'models', 'img', and 'csv' directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('img', exist_ok=True)
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
X = df[features]
y = df['DaysUntilOptimalOrder']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Train RandomForest Model
rf_model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=4)
rf_model.fit(X_train_scaled, y_train)

# 2. Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 3. Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 4. Train Lasso Regression Model
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

# 5. Train Ridge Regression Model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# 6. Train Neural Network Model
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
nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)

# Save the scaler and models
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
joblib.dump(rf_model, os.path.join('models', 'rf_model.pkl'))
joblib.dump(xgb_model, os.path.join('models', 'xgb_model.pkl'))
joblib.dump(lr_model, os.path.join('models', 'lr_model.pkl'))
joblib.dump(lasso_model, os.path.join('models', 'lasso_model.pkl'))
joblib.dump(ridge_model, os.path.join('models', 'ridge_model.pkl'))
nn_model.save(os.path.join('models', 'nn_model.h5'))  # Save model in HDF5 format

print("Models and scaler have been successfully saved in the 'models' folder.")


# # Loading Models and Creating a User Interface

# In[4]:


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




