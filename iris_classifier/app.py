import streamlit as st
import pandas as pd
import joblib
import os

# Function to load the model
def load_model():
    """Load the trained model from the file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'iris_model.joblib')
    model = joblib.load(model_path)
    return model

# Map target integers to species names
species_names = ['setosa', 'versicolor', 'virginica']

def main():
    """Main function to define and run the Streamlit app."""
    st.title("Iris Flower Species Classifier")
    st.write("This app predicts the species of an Iris flower based on its sepal and petal measurements.")

    # Load the trained model
    model = load_model()

    # Sidebar for user input features
    st.sidebar.header("Input Features")

    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    # Display the user input
    st.subheader("User Input:")
    st.write(input_data)

    # Prediction button
    if st.button("Predict"):
        # Make prediction
        prediction = model.predict(input_data)
        predicted_species = species_names[prediction[0]]

        st.subheader("Prediction:")
        st.write(f"The predicted species is **{predicted_species}**.")

if __name__ == '__main__':
    main()
