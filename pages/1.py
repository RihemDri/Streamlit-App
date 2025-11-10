#import module
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Data", "Charts", "Analysis"])
with tab1:
    st.write("### Preview of the Data")
    import pandas as pd 
    import numpy as np
    df = pd.DataFrame(data = np.random.randn(100, 3), columns= ["A", "B", "C"])
    #3 methode pour afficher 
    st.dataframe(df.head(5))
    st.dataframe(df.head())

with tab2:
    selected_feature = st.selectbox("Select feature:", df.columns)
    st.line_chart(df[selected_feature])

with tab3:
    st.write("### Statistical Summary")
    st.write(df.describe())
