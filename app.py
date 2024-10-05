import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved models and encoder
rf_model = joblib.load('insurance_premium_model.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')

# Load the processed DataFrame (df_final) that was used during model training
df_processed = pd.read_pickle('processed_data.pkl')  # Adjust the path to your processed data

# Streamlit app title and layout configuration
st.set_page_config(page_title='Insurance Premium Prediction', layout='wide')

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;  /* Light background */
            color: #343a40;  /* Dark text color */
            font-family: 'Arial', sans-serif;  /* Font family */
        }
        .header {
            text-align: center;
            color: #007BFF;  /* Bright header color */
            font-size: 36px; /* Larger header font size */
            margin-bottom: 20px; /* Spacing below header */
        }
        .stButton {
            background-color: #28a745; /* Button color */
            color: white;  /* Button text color */
            font-weight: bold;
            padding: 10px 20px; /* Padding around button */
            border-radius: 5px; /* Rounded corners */
        }
        .stButton:hover {
            background-color: #218838; /* Darker button hover color */
        }
        .input-label {
            font-weight: bold; /* Bold labels for inputs */
        }
        .help-text {
            font-size: 12px; /* Smaller font for help text */
            color: #6c757d; /* Grey color for help text */
        }
        .prediction {
            background-color: #e9ecef; /* Light grey background for prediction */
            padding: 15px; /* Padding around prediction area */
            border-radius: 5px; /* Rounded corners for prediction box */
            font-size: 24px; /* Larger font size for prediction */
            text-align: center; /* Centered text */
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="header">Insurance Premium Prediction</h1>', unsafe_allow_html=True)

# Collect user inputs in a sidebar
with st.sidebar:
    st.header('Input Features')
    
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
    no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=1)
    
    smoker = st.selectbox('Smoker', ['Yes', 'No'], index=0)
    diabetes = st.selectbox('Diabetes', ['Yes', 'No'], index=0)
    regular_ex = st.selectbox('Regular Exercise', ['Yes', 'No'], index=0)

    # Categorical inputs for encoding (converted to lowercase)
    sex = st.selectbox('Sex', ['Male', 'Female']).lower()  # Convert to lowercase
    hereditary_diseases = st.selectbox('Hereditary Diseases', 
                                       ['NoDisease', 'Alzheimer', 'Arthritis', 'Cancer', 
                                        'Diabetes', 'Epilepsy', 'EyeDisease', 'HeartDisease', 
                                        'High BP', 'Obesity'])

    job_title = st.selectbox('Job Title', 
                             ['Academician', 'Accountant', 'Actor', 'Analyst', 
                              'Architect', 'Beautician', 'Blogger', 'Businessman', 
                              'CA', 'CEO', 'Chef', 'Clerks', 'Dancer', 
                              'DataScientist', 'DefencePersonnels', 'Doctor', 
                              'Engineer', 'Farmer', 'FashionDesigner', 
                              'FilmDirector', 'FilmMaker', 'GovEmployee', 
                              'HomeMakers', 'HouseKeeper', 'ITProfessional', 
                              'Journalist', 'Labourer', 'Lawyer', 'Manager', 
                              'Photographer', 'Police', 'Politician', 
                              'Singer', 'Student', 'Technician'])

    # Tooltips for user guidance
    st.markdown('<p class="help-text">Enter your details to predict the insurance premium.</p>', unsafe_allow_html=True)

# Convert the user inputs into a DataFrame
input_dict = {
    'age': [age],
    'weight': [weight],
    'no_of_dependents': [no_of_dependents],
    'smoker': [1 if smoker == 'Yes' else 0],
    'diabetes': [1 if diabetes == 'Yes' else 0],
    'regular_ex': [1 if regular_ex == 'Yes' else 0],
    'sex': [sex],  # Now using lowercase to match the expected categories
    'hereditary_diseases': [hereditary_diseases],
    'job_title': [job_title]
}
input_df = pd.DataFrame(input_dict)

# Apply One-Hot Encoding to the categorical variables using the saved encoder
categorical_cols = ['sex', 'hereditary_diseases', 'job_title']

try:
    input_encoded = one_hot_encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(input_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    
    # Drop the categorical columns from input_df and combine it with the one-hot encoded columns
    numerical_cols = ['age', 'weight', 'no_of_dependents', 'smoker', 'diabetes', 'regular_ex']
    input_processed = pd.concat([input_df[numerical_cols].reset_index(drop=True), encoded_df], axis=1)
    
    # Ensure the input data columns match the training data format (df_processed), excluding `log_ins_premium`
    df_processed = df_processed.drop(columns=['log_ins_premium'])  # Drop the target column from training data
    
    missing_cols = set(df_processed.columns) - set(input_processed.columns)
    
    # Add missing columns with value 0
    for col in missing_cols:
        input_processed[col] = 0
    
    # Reorder input_processed to match df_processed columns
    input_processed = input_processed[df_processed.columns]
    
    # Prediction button with spinner
    if st.button('Predict'):
        with st.spinner('Calculating...'):
            # Make predictions using the model
            rf_prediction = rf_model.predict(input_processed)

            # Revert the log transformation to get the original insurance premium
            estimated_premium = np.expm1(rf_prediction[0])  # Inverse of log1p to revert to original scale
            
            # Display the prediction
            st.markdown('<div class="prediction">Estimated Premium: **${:,.2f}**</div>'.format(estimated_premium), unsafe_allow_html=True)

except ValueError as e:
    st.error(f"An error occurred during encoding: {e}")
