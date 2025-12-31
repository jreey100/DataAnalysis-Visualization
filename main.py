import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

import streamlit as st

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

def data_info(df):
    # horizontal display of data types
    info = df.dtypes.to_frame().T
    info.columns = df.columns
    info.index = ['']
    return info

def data(df):
    st.write("Statistical Summary:")
    st.write(df.describe())
    st.write(data_info(df))

def scatter_plot_vis(df):
    st.write("DataFrame Columns:")
    st.write(data_info(df))

    x_col = st.text_input("Enter X-axis column name", value="")
    y_col = st.text_input("Enter Y-axis column name", value="")

    # Wait until both names are provided
    if not x_col or not y_col:
        st.info("Enter both column names to generate a scatter plot.")
        return

    # Check that columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.error("One or both column names do not exist in the DataFrame.")
        return

    # Check that both are numeric
    if (not np.issubdtype(df[x_col].dtype, np.number) or
        not np.issubdtype(df[y_col].dtype, np.number)):
        st.write("columns not avaliable to analyze")
        return

    # Drop NaNs from the selected columns only
    clean_df = df[[x_col, y_col]].dropna()

    if clean_df.empty:
        st.write("columns not avaliable to analyze")
        st.info("No valid (non-NaN) rows for the selected columns.")
        return

    st.write(f"Scatter Plot: `{x_col}` vs `{y_col}`")

    fig, ax = plt.subplots()
    sns.regplot(x=x_col, y=y_col, data=clean_df, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot with Regression Line: {x_col} vs {y_col}")

    st.pyplot(fig, use_container_width=True)

def box_plot_vis(df):
    st.write("DataFrame Columns:")
    st.write(data_info(df))

    x_col = st.text_input("Enter categorical X-axis column name", value="")
    y_col = st.text_input("Enter numeric Y-axis column name", value="")

    if not x_col or not y_col:
        st.info("Enter both column names to generate a box plot.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        st.error("One or both column names do not exist in the DataFrame.")
        return

    # Here we require Y to be numeric (X can be anything for grouping)
    if not np.issubdtype(df[y_col].dtype, np.number):
        st.write("columns not avaliable to analyze")
        return

    clean_df = df[[x_col, y_col]].dropna()

    if clean_df.empty:
        st.write("columns not avaliable to analyze")
        st.info("No valid (non-NaN) rows for the selected columns.")
        return

    st.write(f"Box Plot: `{x_col}` vs `{y_col}`")

    fig, ax = plt.subplots()
    sns.boxplot(x=x_col, y=y_col, data=clean_df, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Box Plot: {y_col} by {x_col}")

    st.pyplot(fig, use_container_width=True)

def regression_model(df):
    st.write("DataFrame Columns:")
    st.write(data_info(df))

    feature_col = st.text_input("Enter feature column name (X)", value="")
    target_col = st.text_input("Enter target column name (Y)", value="")

    if not feature_col or not target_col:
        st.info("Enter both column names to fit a regression model.")
        return

    if feature_col not in df.columns or target_col not in df.columns:
        st.error("One or both column names do not exist in the DataFrame.")
        return

    # Both must be numeric for regression
    if (not np.issubdtype(df[feature_col].dtype, np.number) or
        not np.issubdtype(df[target_col].dtype, np.number)):
        st.write("columns not avaliable to analyze")
        return

    clean_df = df[[feature_col, target_col]].dropna()

    if clean_df.empty:
        st.write("columns not avaliable to analyze")
        st.info("No valid (non-NaN) rows for the selected columns.")
        return

    X = clean_df[[feature_col]]
    y = clean_df[[target_col]]

    st.write(f"Linear Regression: `{feature_col}` → `{target_col}`")

    lm = LinearRegression()
    lm.fit(X, y)

    # Coefficient of determination
    st.write("R² Score:", lm.score(X, y))

    # Line for visualization
    X_sorted = np.sort(X.values, axis=0)
    y_line = lm.predict(X_sorted)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.2, label="Data Points")
    ax.plot(X_sorted, y_line, linewidth=1, label="Fitted Line")
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.set_title(f"Linear Regression: {feature_col} vs {target_col}")
    ax.legend()

    # Equation of regression line
    coef = lm.coef_[0][0]
    intercept = lm.intercept_[0]
    st.write("Equation of Regression Line:")
    st.write(f"{target_col} ≈ {intercept:.2f} + {coef:.4f} · {feature_col}")

    st.pyplot(fig, use_container_width=True)

def main():
    st.title("Data Analysis and Visualization App")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Scatter Plot Visualization", "Box Plot Visualization", "Modeling"]
        )

        with tab1:
            st.header("Data Overview")
            data(df)

        with tab2:
            st.header("Scatter Plot Visualization")
            scatter_plot_vis(df)

        with tab3:
            st.header("Box Plot Visualization")
            box_plot_vis(df)

        with tab4:
            st.header("Regression Modeling")
            regression_model(df)

main()
