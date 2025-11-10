import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px 


#page title 
st.title("welcome to my web app ")
#name=st.text_input("enter you name")
#if name :
  #  st.success(f"Hi{name} welcome to my app ")

page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Model"])
if page == "🏠 Home":
  # ------------  KEEP DATA AFTER REFRESH ------------
    if "df" not in st.session_state:
        st.session_state.df = None
        #If there is NO variable named "df" saved in session_state, create it and set it to None.

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

    # if a new file is uploaded ⇒ read & save it
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
    # get the saved dataframe
    df = st.session_state.df



    # ------------  MAIN UI ------------
    if df is not None:
            # ✅ Clear Button here
        if st.button("Clear Data"):
            st.session_state.df = None
            st.rerun()

            
        st.success("File uploaded successfully!")

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

        #display data     # Display selected rows + columns

        st.write(df[:no_rows][choose_col])

        

        # Expander 1 - Summary Statistics
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        st.success("Statistical analysis complete!")


        # ---------------- VISUALIZATION ----------------

        st.subheader("Visualization")


        
        tab1, tab2 = st.tabs(["scatter plot ", "histogram"])
        with tab1:
            #creates a dropdown list.
            col1, col2 = st.columns(2)
            with col1:
             x_col = st.selectbox("Select X-axis", df.columns)
            with col2:
            #ensures the dropdown lists all columns in the dataset.
             y_col = st.selectbox("Select Y-axis", df.columns)

            #Creating a Scatter Plot
            fig_scatter = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
            # Displaying the Plot
            st.plotly_chart(fig_scatter)
        with tab2:

            st.subheader("Histogram")

            hist_col = st.selectbox("Select column for histogram", df.columns)
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)

            fig_hist = px.histogram(df,x=hist_col,nbins=bins, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig_hist)
elif page == "🧠 Model":
 
    st.title("Data Page")
    st.write("This is the data page where you can explore your dataset.")





