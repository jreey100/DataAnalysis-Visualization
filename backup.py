import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

import streamlit as st

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

def data_info(df):
    #horizontal display of data types
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

    # get input from user


    # Display scatter plot
    st.write("Scatter Plot:")
    plot = sns.regplot(x="total_rooms", y="median_house_value", data=df)
    st.pyplot(plot.get_figure(),width=500)

def box_plot_vis(df):
    st.write("DataFrame Columns:")
    st.write(data_info(df))


    # Display box plot
    st.write("Box Plot:")
    boxplot = sns.boxplot(x="ocean_proximity", y="median_house_value", data=df)
    st.pyplot(boxplot.get_figure(),width=500)

def regression_model(df):

    x = df[["total_rooms"]]
    y = df[["median_house_value"]]

    lm = LinearRegression()
    lm.fit(x,y)

    # Coefficent of Correlation
    st.write("R2 Score:", lm.score(x,y))

    # Prediction
    y_pred = lm.predict(x)
    
    fig, ax = plt.subplots()
    ax.scatter(x,y, alpha = 0.2, label="Data Points")

    x_sorted = np.sort(x.values, axis=0)
    y_line = lm.predict(x_sorted)

    ax.plot(x_sorted, y_line, linewidth=1, label="Fitted Line", color='red')
    ax.set_xlabel("Total rooms")
    ax.set_ylabel("Median house value")
    ax.set_title("Linear Regression: total_rooms vs median_house_value")
    ax.legend()

    st.write("Equation of Regression Line:")
    coef = lm.coef_[0][0]
    intercept = lm.intercept_[0]
    st.write(f"median_house_value ≈ {intercept:.2f} + {coef:.4f} · total_rooms")

    st.pyplot(fig, width = 500)

    
def main():
    st.title("Data Analysis and Visualization App")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Scatter Plot Visulization", "Box Plot Visulization", "Modeling"])

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