#import module
import streamlit as st
import pandas as pd
import numpy as np


#Title
st.title("Hello Streamlit!")    #: Adds a big title

#Header and Subheader:

st.header("Welcome to Your First Streamlit App")  #: Adds a section header
st.subheader("This is a subheader")

#Text
st.text("This is your first interactive data app using Python!")    #: Displays text
st.text_area('Description') #Description area 

##########"
st.success("yesssssssssssss!!!")
st.error("something  went wrong")
st.info('info')
st.warning("be carful !!")
#############

st.date_input("Your birthday") 
st.time_input("Meeting time")
st.number_input("Pick a number", 0, 10)
############


# Audio
st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")

# Video
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


##########
# Slider
score = st.slider(
    "Rate your experience",
    min_value=0,
    max_value=10,
    value=5
)

# Range slider
values = st.slider(
    "Select a range",
    min_value=0.0,
    max_value=100.0,
    value=(25.0, 75.0)
)


# Add to sidebar
st.sidebar.title("Navigation")
#page = st.sidebar.selectbox("Choose a page", ["Home", "About", "Contact"])

# Sidebar inputs
#name = st.sidebar.text_input("Your name")
#age = st.sidebar.slider("Your age", 0, 100)

# Display based on sidebar selection
#if page == "Home":
   # st.title("Home Page")
   # st.write(f"Welcome, {name}! You are {age} years old.")
#elif page == "About":
 #   st.title("About Page")
#elif page == "Contact":
 #   st.title("Contact Page")


#############
name =st.text_input("enter you name ")
if (st.button("submit")):
    st.success(f"hello,{name}")

st.date_input("enter ur birth ")
st.time_input("meeting time ")


# DATAFRAME

df=pd.DataFrame(data=np.random.randn(100,3),columns=["A","B","c"])
a =st.sidebar.number_input("pick a number",min_value=1,max_value=len(df))
st.dataframe(df.head(a))


evel = st.sidebar.slider("Choose a level", min_value=1, max_value=500)
# Display the selected level
 
st.dataframe(df.head(evel))



st.title("map")
# Tunisia approximate coordinates (centered near Tunis)
tunisia_lat = 36.8 # latitude
tunisia_lon = 10.2 # longitude

# Create random data around Tunisia
map_data = pd.DataFrame({ 
  'lat':[33.57,36.8], 'lon':[-7.67,10.18]}
)

# Display the map
st.map(map_data, zoom=6)
 
