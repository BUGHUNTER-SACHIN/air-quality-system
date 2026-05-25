import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Sentioair AQI Intelligence System",
    page_icon="🌬️",
    layout="wide"
)

# Title
st.title("🌬️ Intelligent Air Quality Monitoring System")
st.markdown("** AI powered AQI Prediction & Anomaly Detection**")
st.divider()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/city_day.csv')
    df = df.drop(columns=['Xylene'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna(subset=['AQI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year'] = df['Date'].dt.year
    return df

df = load_data()

# Sidebar
st.sidebar.title("⚙️ Controls")
cities = sorted(df['City'].unique())
selected_city = st.sidebar.selectbox("Select City", cities, index=cities.index('Delhi'))

# Filter city data
city_df = df[df['City'] == selected_city].sort_values('Date').reset_index(drop=True)

# --- SECTION 1: City Overview ---
st.header(f"📍 {selected_city} — Air Quality Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average AQI", f"{city_df['AQI'].mean():.0f}")
col2.metric("Max AQI", f"{city_df['AQI'].max():.0f}")
col3.metric("Min AQI", f"{city_df['AQI'].min():.0f}")
col4.metric("Total Days", f"{len(city_df)}")

# AQI over time
st.subheader("📈 AQI Over Time")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(city_df['Date'], city_df['AQI'], color='royalblue', linewidth=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.set_title(f"{selected_city} AQI Over Time")
st.pyplot(fig)

# --- SECTION 2: Random Forest Prediction ---
st.header("🌲 AQI Prediction — Random Forest")

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
            'NH3', 'CO', 'SO2', 'O3', 'Benzene',
            'Toluene', 'Month', 'Day', 'Year']

X = df[features]
y = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with st.spinner("Training Random Forest..."):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)

col1, col2 = st.columns(2)
col1.metric("Model Accuracy (R²)", f"{r2*100:.1f}%")
col2.metric("Mean Absolute Error", f"{mae:.1f} AQI pts")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True).tail(8)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='crimson')
ax.set_title("Top Features Affecting AQI")
st.pyplot(fig)

# --- SECTION 3: Anomaly Detection ---
st.header("🚨 Anomaly Detection — Isolation Forest")

anomaly_features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
X_city = city_df[anomaly_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_city)

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
city_df['anomaly'] = iso_forest.fit_predict(X_scaled)
anomalies = city_df[city_df['anomaly'] == -1]

st.metric("Anomalous Days Detected", f"{len(anomalies)} days")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(city_df['Date'], city_df['AQI'], color='blue', linewidth=0.8, label='Normal')
ax.scatter(anomalies['Date'], anomalies['AQI'], color='red', s=40, zorder=5, label='⚠️ Anomaly')
ax.set_title(f"{selected_city} — Anomaly Detection")
ax.legend()
st.pyplot(fig)

# Top anomalies table
st.subheader("🔴 Most Dangerous Days")
top = anomalies.sort_values('AQI', ascending=False)[['Date', 'AQI', 'PM2.5', 'CO']].head(5)
st.dataframe(top, use_container_width=True)

st.divider()
st.markdown("*Built by Sachin Singh | *")