import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="House Price Prediction App", page_icon="🏡", layout="wide"
)


# Load the trained model pipeline
@st.cache_resource
def load_model():
  return joblib.load("best_house_price_model.pkl")


# Load dataset to extract categorical options and ranges dynamically
@st.cache_data
def load_data():
  return pd.read_csv("house_price_50k.csv")


model = load_model()
df = load_data()

st.title("🏡 House Price Prediction App")
st.write(
    "Enter the property details below to get an estimated house price based"
    " on your trained Ridge regression model."
)

st.markdown("---")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
  st.subheader("Property Dimensions & Layout")
  area = st.number_input(
      "Area (sq ft)",
      min_value=int(df["area"].min()),
      max_value=int(df["area"].max()),
      value=int(df["area"].mean()),
  )
  bedrooms = st.slider(
      "Bedrooms",
      int(df["bedrooms"].min()),
      int(df["bedrooms"].max()),
      int(df["bedrooms"].median()),
  )
  bathrooms = st.slider(
      "Bathrooms",
      int(df["bathrooms"].min()),
      int(df["bathrooms"].max()),
      int(df["bathrooms"].median()),
  )
  floors = st.slider(
      "Floors",
      int(df["floors"].min()),
      int(df["floors"].max()),
      int(df["floors"].median()),
  )
  age = st.number_input(
      "Age of Property (Years)",
      float(df["age"].min()),
      float(df["age"].max()),
      float(df["age"].mean()),
  )
  distance = st.number_input(
      "Distance to City Center",
      float(df["distance"].min()),
      float(df["distance"].max()),
      float(df["distance"].mean()),
  )
  garage = st.slider(
      "Garage Spaces",
      int(df["garage"].min()),
      int(df["garage"].max()),
      0,
  )
  parking = st.slider(
      "Parking Spaces",
      int(df["parking"].min()),
      int(df["parking"].max()),
      0,
  )

with col2:
  st.subheader("Amenities & Neighborhood")
  garden = st.selectbox("Garden Available?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  security = st.selectbox("Security Available?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  school_nearby = st.selectbox("School Nearby?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  hospital_nearby = st.selectbox("Hospital Nearby?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  shopping_mall_nearby = st.selectbox("Shopping Mall Nearby?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  public_transport = st.selectbox("Public Transport Nearby?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
  crime_rate = st.number_input(
      "Crime Rate",
      float(df["crime_rate"].min()),
      float(df["crime_rate"].max()),
      float(df["crime_rate"].mean()),
  )
  population_density = st.number_input(
      "Population Density",
      float(df["population_density"].min()),
      float(df["population_density"].max()),
      float(df["population_density"].mean()),
  )

st.markdown("---")
st.subheader("Location & Demographics")

col3, col4 = st.columns(2)
with col3:
  # Extract unique categories from dataset
  location_options = df["location"].unique().tolist()
  location = st.selectbox("Location Type", location_options)

with col4:
  income_options = df["income_level"].unique().tolist()
  income_level = st.selectbox("Income Level", income_options)

# Prediction Logic
if st.button("Predict Price", type="primary"):
  # Construct a DataFrame matching the input structure of the pipeline
  input_data = pd.DataFrame({
      "area": [area],
      "bedrooms": [bedrooms],
      "bathrooms": [bathrooms],
      "floors": [floors],
      "age": [age],
      "distance": [distance],
      "garage": [garage],
      "parking": [parking],
      "garden": [garden],
      "security": [security],
      "school_nearby": [school_nearby],
      "hospital_nearby": [hospital_nearby],
      "shopping_mall_nearby": [shopping_mall_nearby],
      "public_transport": [public_transport],
      "crime_rate": [crime_rate],
      "population_density": [population_density],
      "location": [location],
      "income_level": [income_level],
  })

  try:
    prediction = model.predict(input_data)
    st.success(f"### Estimated House Price: **Pkr {prediction[0]:,.2f}**")
  except Exception as e:
    st.error(f"Prediction failed due to: {e}")