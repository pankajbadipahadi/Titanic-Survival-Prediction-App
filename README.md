# 🚢 Titanic Survival Prediction App  

A **machine learning web application** built with [Streamlit](https://streamlit.io/) that predicts whether a Titanic passenger would survive, based on their details.  
The model behind the app is a **Logistic Regression classifier** trained on the famous [Titanic dataset].  

---

## 📌 Features
✅ User-friendly web interface powered by Streamlit  
✅ Real-time predictions of survival probability  
✅ Encodes categorical inputs automatically (`Sex`, `Embarked`)  
✅ Can run **locally** with Anaconda or be **deployed online** with Streamlit Cloud  
✅ Modular design – you can replace the model with any `.pkl` file trained on the same features  

---

## 🧠 Machine Learning Model
The model was trained using **Logistic Regression** with `scikit-learn`.  

### Features used for training:
- `Pclass` – Passenger class (1st, 2nd, 3rd)  
- `Sex` – Male/Female  
- `Age` – Passenger age  
- `SibSp` – Number of siblings/spouses aboard  
- `Parch` – Number of parents/children aboard  
- `Fare` – Ticket fare  
- `Embarked` – Port of Embarkation (`C`, `Q`, `S`)  

Categorical features were encoded using **OneHotEncoder / pd.get_dummies** before training.  
The model outputs:  
- **0 → Not Survive**  
- **1 → Survive**  

---

## 📂 Project Structure
