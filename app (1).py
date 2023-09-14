import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import gradio as gr

# Load the trained SVM model
svm_model = SVC(kernel='linear', C=1)  # Load your trained model here

# Define categorical columns for one-hot encoding
categorical_columns = ['Past Medical History', 'Family History with Disease', 'Allergies', 'Sex', 'Pregnant', 'Currently Menstruating']

def preprocess_input(input_data):
    # Convert categorical variables to numerical representations
    input_data = pd.get_dummies(input_data, columns=categorical_columns)
    
    # Standardize numerical features
    scaler = StandardScaler()
    input_data[['Weight (kg)', 'Height (cm)', 'Duration of Disease (days)', 'Age']] = scaler.fit_transform(input_data[['Weight (kg)', 'Height (cm)', 'Duration of Disease (days)', 'Age']])
    
    return input_data

def predict_disease(input_data):
    # Preprocess input data
    input_data = preprocess_input(pd.DataFrame([input_data]))  # Wrap input_data in a DataFrame
    
    # Predict disease
    prediction = svm_model.predict(input_data)[0]
    
    return prediction

# Define Gradio interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.inputs.Textbox(label="Name"),
        gr.inputs.Radio(choices=["Disease1", "Disease2", "Disease3"], label="Select Disease"),
        gr.inputs.Textbox(label="Symptom 1"),
        gr.inputs.Textbox(label="Symptom 2"),
        gr.inputs.Textbox(label="Symptom 3"),
        gr.inputs.Number(label="Weight (kg)"),
        gr.inputs.Number(label="Height (cm)"),
        gr.inputs.Textbox(label="Past Medical History"),
        gr.inputs.Textbox(label="Family History with Disease"),
        gr.inputs.Textbox(label="Allergies"),
        gr.inputs.Number(label="Duration of Disease (days)"),
        gr.inputs.Number(label="Age"),
        gr.inputs.Radio(choices=["Male", "Female"], label="Sex"),
        gr.inputs.Radio(choices=["Yes", "No"], label="Pregnant"),
        gr.inputs.Number(label="Number of Times Pregnant"),
        gr.inputs.Radio(choices=["Yes", "No"], label="Currently Menstruating"),
    ],
    outputs=gr.outputs.Label(label="Predicted Disease"),
    title="Disease Prediction SVM Model",
    description="Enter patient information to predict the disease.",
)

# Launch the Gradio app
iface.launch()
