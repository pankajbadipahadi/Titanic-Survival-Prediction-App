import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model and Preprocessing Objects ---
try:
    with open("titanic_model.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    model = loaded_data['model']
    scaler = loaded_data['scaler']
    le = loaded_data['label_encoders']
    features = loaded_data['features']

except FileNotFoundError:
    st.error("Error: 'titanic_model.pkl' not found. Ensure the model file is in the same directory.")
    st.stop()
except KeyError:
    st.error("Error: Model file does not contain expected keys ('model', 'scaler', 'label_encoders', 'features').")
    st.stop()

# --- 2. Define Outlier Bounds (MUST BE COPIED FROM TRAINING NOTEBOOK) ---
# NOTE: These values must be obtained by running the IQR calculation on the *original*
# Titanic_train.csv and saving the lower/upper bounds for each column.
# The exact bounds depend on the data distribution. The values below are educated guesses.
# You must replace these with the exact bounds calculated in lr_assignment.py!
OUTLIER_BOUNDS = {
    'Age': {'lower': 2.0, 'upper': 60.0},     # e.g., Q1-1.5*IQR and Q3+1.5*IQR from training
    'SibSp': {'lower': 0.0, 'upper': 3.0},
    'Parch': {'lower': 0.0, 'upper': 0.0},
    'Fare': {'lower': 0.0, 'upper': 65.65}
}

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")
st.write("Enter passenger details to predict survival.")

# --- Input fields ---
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- Convert inputs into DataFrame ---
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

st.write("### Input Data Preview")
st.dataframe(input_data)

# --- Prediction ---
if st.button("Predict"):
    try:
        # --- 3. REPLICATE PREPROCESSING STEPS ---
        
        # A. Label Encoding for Sex and Embarked
        # WARNING: Your training code reused the same 'le' for both features, which is bad practice.
        # We will manually map based on the common Titanic dataset encoding:
        # Sex: 'male' -> 1, 'female' -> 0 (Check your training data for this exact mapping)
        # Embarked: 'S'->2, 'C'->0, 'Q'->1 (Check the saved 'le' for this mapping)
        
        # You need to determine the correct mappings from your trained 'le' object, 
        # but for simplicity, we use the values 'le' holds after being fitted to 'Embarked'
        
        # To avoid relying on the faulty reusable 'le', we use explicit mappings
        sex_mapping = {'male': 1, 'female': 0}
        embarked_mapping = {'S': 2, 'C': 0, 'Q': 1} # This order is an educated guess based on Pclass/Sex being more important

        input_data['Sex'] = input_data['Sex'].map(sex_mapping)
        # Handle Embarked safely: fill NaNs (if any) with the mode (which was 'S' -> 2 in training)
        input_data['Embarked'] = input_data['Embarked'].map(embarked_mapping)
        
        
        # B. Outlier Capping (Critical step from training)
        for col, bounds in OUTLIER_BOUNDS.items():
            lower = bounds['lower']
            upper = bounds['upper']
            input_data[col] = np.where(input_data[col] < lower, lower, input_data[col])
            input_data[col] = np.where(input_data[col] > upper, upper, input_data[col])
            
        # C. Select Features and Scaling
        # Select features in the correct order: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        input_for_scaling = input_data[features] 
        
        # Scale the data using the *fitted* scaler
        input_scaled = scaler.transform(input_for_scaling)
        
        # 🔮 Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # --- Display Results ---
        if prediction == 1:
            st.success(f"🎉 The passenger is predicted to **Survive** with probability {probability:.2f}")
        else:
            st.error(f"☠️ The passenger is predicted **Not to Survive** with probability {1-probability:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Check your feature columns, mappings, and loaded objects.")