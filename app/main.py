import streamlit as st
import pandas as pd

# -------------------------------------
# App Configuration
# -------------------------------------
st.set_page_config(
    page_title="Effex — Causal ML App",
    layout="wide"
)

st.title("📊 Effex — Causal ML for Policy Analysis")
st.write("Upload a dataset to begin.")


# -------------------------------------
# Data Loader (with caching)
# -------------------------------------
@st.cache_data
def load_data(file):
    ext = file.name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(file)

    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(file)  # requires openpyxl installed

    else:
        raise ValueError("Unsupported file format. Only CSV, XLS, XLSX allowed.")


# -------------------------------------
# File Upload Widget
# -------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)


# -------------------------------------
# If File Uploaded → Process Dataset
# -------------------------------------
if uploaded_file:

    st.success("File uploaded successfully! ✔")

    try:
        df = load_data(uploaded_file)

        # -------------------------------
        # Dataset Preview
        # -------------------------------
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------------------
        # Summary Metric Cards
        # -------------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        # -------------------------------
        # Column Types
        # -------------------------------
        st.subheader("🧬 Column Types")
        st.write(df.dtypes)

        # -------------------------------
        # Missing Values per Column
        # -------------------------------
        st.subheader("⚠ Missing Values by Column")
        st.dataframe(df.isnull().sum(), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")


# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.caption("Effex — Causal ML Toolkit for Policy Analysis")
