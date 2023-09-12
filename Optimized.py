import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Set Plotly template to white background
import plotly.io as pio
pio.templates.default = "plotly_white"

# Load data
data = pd.read_csv('retail_price.csv')

# Streamlit app
st.title('Retail Price Analysis App')


st.sidebar.header('Visualization Options')
chart_options = {
    'Histogram': 'Distribution of Total Price',
    'Box Plot': 'Box Plot of Unit Price',
    'Scatter Plot': 'Quantity vs Total Price with Trendline',
    'Bar Chart': 'Average Total Price by Product Category',
    'Correlation Heatmap': 'Correlation Heatmap of Numerical Features',
    'Bar Chart - Price Difference': 'Average Competitor Price Difference by Product Category'
}
selected_chart = st.sidebar.selectbox('Select a visualization:', list(chart_options.keys()))

# Visualization
st.header(chart_options[selected_chart])
if selected_chart == 'Histogram':
    fig = px.histogram(data, x='total_price', nbins=20)
elif selected_chart == 'Box Plot':
    fig = px.box(data, y='unit_price')
elif selected_chart == 'Scatter Plot':
    fig = px.scatter(data, x='qty', y='total_price', trendline='ols')
elif selected_chart == 'Bar Chart':
    avg_price_by_category = data.groupby('product_category_name')['total_price'].mean().reset_index()
    fig = px.bar(avg_price_by_category, x='product_category_name', y='total_price')
elif selected_chart == 'Correlation Heatmap':
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    fig = go.Figure(go.Heatmap(x=correlation_matrix.columns, y=correlation_matrix.columns, z=correlation_matrix.values))
elif selected_chart == 'Bar Chart - Price Difference':
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']
    avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
    fig = px.bar(avg_price_diff_by_category, x='product_category_name', y='comp_price_diff')
st.plotly_chart(fig)

# Sidebar - Modeling Options
st.sidebar.header('Modeling Options')
model_button = st.sidebar.button('Train Decision Tree Regressor Model')

# Model Training and Display Predictions
if model_button:
    st.header('Predicted vs Actual Retail Price')
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']
    
    X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
    y = data['total_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='blue'), name='Predicted vs. Actual'))
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', marker=dict(color='red'), name='Ideal Prediction'))
    fig.update_layout(title='Predicted vs Actual Retail Price', xaxis_title='Actual Retail Price', yaxis_title='Predicted Retail Price')
    st.plotly_chart(fig)
    
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')
