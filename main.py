import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px 


#page title 
st.title("welcome to my web app ")
name=st.text_input("enter you name")
#if name :
  #  st.success(f"Hi{name} welcome to my app ")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if st.checkbox("show data"):
        st.dataframe(df.head()) # Display data
    #st.table(df) # Static table
    
    
        # Optional:we can add for data
    with st.expander("🔍 Data Preview"):
        st.dataframe(df.head())


        
    st.write("Shape:", df.shape) # prints this information to the app.
    st.write("Columns:", df.columns.tolist()) # lists all column names in the dataset

    #slider  
    no_rows =st.slider("select rows ", min_value=1, max_value=len(df))
    
    #multi-select box 
    choose_col = st.multiselect("Select columns to show :", df.columns.to_list(),default=df.columns.to_list())

    #display data 
    st.write(df[:no_rows][choose_col])

     

     # Expander 1 - Summary Statistics
    with st.expander("Summary Statistics"):
        st.write(df.describe())

    st.success("Statistical analysis complete!")



    st.subheader("viz")


      
    #creates a dropdown list.
    x_col = st.selectbox("Select X-axis", df.columns)
    #ensures the dropdown lists all columns in the dataset.
    y_col = st.selectbox("Select Y-axis", df.columns)

    #Creating a Scatter Plot
    fig_scatter = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
    # Displaying the Plot
    st.plotly_chart(fig_scatter)






