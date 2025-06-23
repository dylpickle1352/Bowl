
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load your CSV file
df = pd.read_csv("New Ball chart - Sheet1.csv")

# Clean data
df = df[['Company', 'Ball Name', 'Trq', 'Len', 'Back', 'Hook']].dropna()

# Model: Predict Hook
X = df[['Trq', 'Len', 'Back']]
y = df['Hook']
model = LinearRegression().fit(X, y)
df['Predicted Hook'] = model.predict(X)

# Streamlit App
st.title("Interactive Bowling Ball Hook Chart")

# Plotly scatter plot
fig = px.scatter(
    df,
    x="Hook",
    y="Predicted Hook",
    hover_data=["Company", "Ball Name", "Trq", "Len", "Back"],
    title="Actual vs Predicted Hook (Hover to see Ball Info)",
)

st.plotly_chart(fig)

# Instructions
st.markdown("### Explore Ball Details Below")
st.markdown("""
- Hover to view ball name and stats.
- Use box zoom or lasso tool to explore.
""")
