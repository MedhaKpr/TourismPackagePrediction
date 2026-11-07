import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline
model_path = os.path.join(os.path.dirname(__file__), "../model_building/best_model.pkl")
if not os.path.exists(model_path):
    # Fallback in case of flat file structures during deployment
    model_path = "best_model.pkl"

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the predictive model: {e}")

# Streamlit UI Configuration
st.set_page_config(page_title="Wellness Tourism Purchase Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.write("""
This interactive tool predicts whether a customer will purchase the newly introduced **Wellness Tourism Package** 
based on their demographics and salesperson interaction characteristics.
""")

# Create two columns for user inputs
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Customer Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (INR)", min_value=1000, max_value=200000, value=25000, step=500)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)

with col2:
    st.header("📞 Sales Interaction")
    typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15, step=1)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
    num_trips = st.number_input("Number of Annual Trips", min_value=0, max_value=20, value=2, step=1)
    pref_property_star = st.slider("Preferred Hotel Rating (Stars)", min_value=3, max_value=5, value=3, step=1)

# Extra demographic fields
st.subheader("🚗 Additional Information")
col3, col4, col5 = st.columns(3)
with col3:
    passport = st.selectbox("Has Valid Passport?", ["Yes", "No"])
with col4:
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
with col5:
    num_people_visiting = st.number_input("Total People Accompanying", min_value=1, max_value=10, value=2, step=1)

num_children_visiting = st.number_input("Number of Children Accompanying (Under Age 5)", min_value=0, max_value=5, value=0, step=1)

# Map input values to match model training features
passport_numeric = 1 if passport == "Yes" else 0
own_car_numeric = 1 if own_car == "Yes" else 0

input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_people_visiting,
    'PreferredPropertyStar': pref_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport_numeric,
    'OwnCar': own_car_numeric,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups
}])

# Prediction Action
st.markdown("--- ")
if st.button("🔮 Predict Purchase Probability", use_container_width=True):
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"🎉 **High Likelihood of Purchase!** This customer is highly likely to purchase the Wellness Package. (Confidence: **{prediction_proba * 100:.2f}%**)")
        else:
            st.warning(f"⚠️ **Low Likelihood of Purchase.** This customer is unlikely to purchase the Wellness Package. (Confidence: **{(1 - prediction_proba) * 100:.2f}%**)")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
