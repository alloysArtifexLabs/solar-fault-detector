import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and label map
model = joblib.load("fault_model.pkl")
label_map = {0: "Disconnected", 1: "Normal", 2: "Overvoltage", 3: "Underperforming"}

# Title and instructions
st.title("⚡ AI-Powered Solar Fault Detector")
st.markdown("Upload your solar panel sensor data to predict faults using AI.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Sensor Data")
    st.dataframe(df.head())

    # Make predictions
    features = df[["Voltage(V)", "Current(A)", "Power(W)"]]
    predictions = model.predict(features)
    df["Predicted Status"] = [label_map[p] for p in predictions]

    st.subheader("✅ Fault Prediction Results")
    st.dataframe(df[["Voltage(V)", "Current(A)", "Power(W)", "Predicted Status"]])

    # ✨ CSV Download Button
    import io
    csv = df[["Voltage(V)", "Current(A)", "Power(W)", "Predicted Status"]].to_csv(index=False)
    b = io.BytesIO()
    b.write(csv.encode())
    b.seek(0)

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=b,
        file_name="solar_fault_predictions.csv",
        mime="text/csv"
    )

    # Plotting charts
    st.subheader("📈 Visualizations")
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(df["Voltage(V)"])
    axs[0].set_title("Voltage Over Time")
    axs[1].plot(df["Current(A)"])
    axs[1].set_title("Current Over Time")
    axs[2].plot(df["Power(W)"])
    axs[2].set_title("Power Over Time")
    st.pyplot(fig)
