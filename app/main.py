import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Effex — Causal ML App",
    layout="wide"
)

st.title("📊 Effex — Causal ML for Policy Analysis")
st.write("Upload a dataset to begin.")

@st.cache_data
def load_data(file):
    ext = file.name.split(".")[-1]
    if ext == "csv":
        return pd.read_csv(file)
    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    st.success("File uploaded successfully!")

    try:
        df = load_data(uploaded_file)

        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📐 Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("🧬 Column Types")
        st.write(df.dtypes)

        st.subheader("⚠ Missing Values")
        st.write(df.isnull().sum())

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
