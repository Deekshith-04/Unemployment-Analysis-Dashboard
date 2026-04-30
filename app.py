import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Unemployment Dashboard", layout="wide")

st.title("📊 Unemployment Analysis Dashboard")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation'
}, inplace=True)

df = df.dropna()

# COVID DATA
df_covid = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df_covid.columns = df_covid.columns.str.strip()
df_covid['Date'] = pd.to_datetime(df_covid['Date'], dayfirst=True)

df_covid.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation'
}, inplace=True)

df_covid = df_covid.dropna()

# MODEL
model = joblib.load("unemployment.pkl")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔎 Filters")

state = st.sidebar.selectbox("Select State", sorted(df['Region'].unique()))
area = st.sidebar.selectbox("Select Area", df['Area'].unique())

filtered_df = df[(df['Region'] == state) & (df['Area'] == area)]

# ------------------ KPIs ------------------
st.subheader("📌 Key Metrics")

k1, k2, k3 = st.columns(3)

k1.metric("Avg Unemployment (%)", round(filtered_df['Unemployment_Rate'].mean(), 2))
k2.metric("Max Unemployment (%)", round(filtered_df['Unemployment_Rate'].max(), 2))
k3.metric("Avg Employment", f"{int(filtered_df['Employed'].mean()):,}")

st.markdown("---")

# ------------------ GRAPH ROW 1 ------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("📈 Unemployment Trend")
    fig, ax = plt.subplots(figsize=(6,3))
    filtered_df.groupby('Date')['Unemployment_Rate'].mean().plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Rate (%)")
    ax.grid()
    st.pyplot(fig, use_container_width=True)

with g2:
    st.subheader("🌆 Rural vs Urban")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    sns.barplot(data=df, x='Area', y='Unemployment_Rate', ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("Rate (%)")
    st.pyplot(fig2, use_container_width=True)

st.markdown(" ")

# ------------------ GRAPH ROW 2 ------------------
g3, g4 = st.columns(2)

with g3:
    st.subheader("🏆 Top 10 States")
    top_states = df.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.barplot(x=top_states.values, y=top_states.index, ax=ax3)
    ax3.set_xlabel("Rate (%)")
    st.pyplot(fig3, use_container_width=True)

with g4:
    st.subheader("🦠 COVID Impact")
    fig4, ax4 = plt.subplots(figsize=(6,3))
    df.groupby('Date')['Unemployment_Rate'].mean().plot(ax=ax4, label="Before")
    df_covid.groupby('Date')['Unemployment_Rate'].mean().plot(ax=ax4, label="During")
    ax4.legend()
    ax4.set_xlabel("")
    ax4.set_ylabel("Rate (%)")
    ax4.grid()
    st.pyplot(fig4, use_container_width=True)

st.markdown("---")

# ------------------ PREDICTION ------------------
st.subheader("🤖 Unemployment Prediction")

col1, col2 = st.columns(2)

with col1:
    emp = st.slider("Estimated Employed", 0, 50000000, 10000000, step=1000000)

with col2:
    labour = st.slider("Labour Participation Rate (%)", 10.0, 70.0, 40.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[emp, labour]], columns=['Employed', 'Labour_Participation'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Unemployment Rate: {prediction[0]:.2f}%")