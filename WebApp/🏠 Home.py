import streamlit as st
from pathlib import Path

path = Path.cwd()

col1, col2, col3 = st.columns(3)
with col2:
    st.image(str(path.joinpath("Logo.png")), use_column_width=True)

st.title("CPP")
st.header("An ML model that predicts car prices...")

st.container(height=10, border=False)

st.link_button(label="Click here to start model training process !",
               url="http://localhost:8501/Modeling",
               type="primary",
               use_container_width=True)

st.divider()

st.subheader("What can i do with the model responses?")
st.write("""
        if you want to buy a car or sell your car, you can use our model to estimate your
        car value and then can make a more informed decision.
        """)

st.container(height=5, border=False)

st.subheader("Main features of the model:")
st.write("""
        🔹 95% accuracy and higher...\n
        🔹 Predicting car prices from Audi, Ford, Toyota, BMW, Mercedes Benz, and ...\n
        🔹 More than 97,000 sample...\n
        🔹 Predicting based on the most importan features like model, year, enginesize,
           fueltype, and ...\n
        🔹 Using an non-deep ML algorithm and quick training time... (relative to Neural
           Network algorithms)\n
        🔹 Simple GUI and good UX while working with the model...\n
        🔹 And so on...\n
        """)

st.container(height=5, border=False)

st.subheader('Surprised and excited? Lets go further!')
st.write('Click on the "👋Introduction" in the sidebar...')
