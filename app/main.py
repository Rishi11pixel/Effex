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
                # ---------------------------------
        # Phase 2: Variable Role Selection
        # ---------------------------------
        st.subheader("🎯 Select Variable Roles (Treatment, Outcome, Confounders)")

        columns = list(df.columns)

        with st.expander("🧪 Treatment Variable"):
            treatment = st.selectbox("Select the treatment variable (T):", columns)

        with st.expander("📈 Outcome Variable"):
            outcome = st.selectbox("Select the outcome variable (Y):", columns)

        with st.expander("🧩 Confounders (X)"):
            confounders = st.multiselect(
                "Select confounder variables:",
                [col for col in columns if col not in [treatment, outcome]]
            )

        with st.expander("⚙ Optional Controls / Covariates"):
            controls = st.multiselect(
                "Select additional control variables (optional):",
                [col for col in columns if col not in [treatment, outcome] + confounders]
            )

        with st.expander("⏳ Time Variable (Optional)"):
            time_var = st.selectbox(
                "Select time variable (if applicable):",
                ["None"] + columns
            )

        # ---------------------------------
        # Validation Rules
        # ---------------------------------
        if treatment == outcome:
            st.error("❌ Treatment and outcome cannot be the same variable.")

        if set(confounders) & {treatment, outcome}:
            st.error("❌ Confounders cannot include treatment or outcome.")

        if st.button("Confirm Variable Selection"):
            if treatment == outcome:
                st.error("Fix errors before proceeding.")
            else:
                st.success("Variables selected successfully!")
                # ---------------------------------
        # Phase 3: Build Causal Graph (DAG)
        # ---------------------------------
        st.subheader("🧠 Causal Graph (DAG)")

        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # add nodes
        G.add_node(treatment)
        G.add_node(outcome)
        for c in confounders:
            G.add_node(c)
        for c in controls:
            G.add_node(c)

        # add edges
        for c in confounders:
            G.add_edge(c, treatment)
            G.add_edge(c, outcome)

        for c in controls:
            G.add_edge(c, outcome)

        G.add_edge(treatment, outcome)

        # draw graph
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx(G, with_labels=True, node_color="skyblue", node_size=1400, font_size=10)
        st.pyplot(fig)

        # ---------------------------------
        # Show Causal Summary
        # ---------------------------------
        st.subheader("📌 Causal Structure Summary")

        st.markdown(f"""
        - **Treatment (T):** `{treatment}`
        - **Outcome (Y):** `{outcome}`
        - **Confounders:** `{confounders}`
        - **Controls:** `{controls}`
        """)


    except Exception as e:
        st.error(f"Error loading file: {str(e)}")


# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.caption("Effex — Causal ML Toolkit for Policy Analysis")
