import streamlit
import joblib

model=joblib.load("lregression.pkl")

Year=int(input("Enter Year "))

output=model.predict([[Year]])

print(f"Import Value for {Year}: {output[0]}")
