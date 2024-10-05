import streamlit as st
import joblib
import numpy as np
# Load the trained model
sales_model = joblib.load('random_sales_detection.pkl')




def main():
    # Streamlit app title
    st.title('Sales Detection Model')

    html_temp = """
    <div style="background-color:blue; padding:10px">
    <h3 style="color:white; text-align:center;">Sales Prediction means predicting how much of product people will buy based on factors such as amount you spend to advertise your product, the segment of people you advertise for, or the platform you used for the advertisement. </h3>
    </div>

    """

    st.markdown(html_temp, unsafe_allow_html=True)        
    # Input Text for Tv Sales
    tv_input = st.number_input('Amount spent on TV Advert', min_value=0.0, max_value=500.0, value=23.0)
    # Input Text for Radio Sales
    radio_input = st.number_input('Amount spent on Radio Advert', min_value=0.0, max_value=150.0, value=23.0)
    # Input Text for Tv Sales
    news_input = st.number_input('Amount spent on NewsPaper Advert', min_value=0.0, max_value=600.0, value=20.0)


    # Predict button

    if st.button('Predict'):
        features = np.array([[tv_input, radio_input, news_input]])
        prediction = sales_model.predict(features)
    
        st.success(f"Predicted Sales from the Adverts: {prediction[0]}")   


        html_temp = """
        <div style="background-color:black; padding:10px"; color:white;>
        <h5 style="color:white; text-align:center;">&copy 2024 Sales Prediction Model trained by: Odeyale Kehinde Musiliudeen </h5>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()