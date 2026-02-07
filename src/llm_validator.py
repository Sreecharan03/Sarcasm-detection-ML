import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def validate_sarcasm(text, ml_prediction):
    """
    Validates sarcasm using the Gemini 2.0 Flash model.

    Args:
        text: The input text to analyze.
        ml_prediction: The prediction from the existing ML model.

    Returns:
        A string indicating the Gemini model's validation result.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"The following text is predicted as '{ml_prediction}' by a machine learning model. Please validate if the text is sarcastic or not. Text: '{text}'"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    # For testing purposes
    test_text = "I'm so excited to go to the dentist."
    test_prediction = "Sarcastic"
    validation_result = validate_sarcasm(test_text, test_prediction)
    print(f"Text: {test_text}")
    print(f"ML Prediction: {test_prediction}")
    print(f"Gemini Validation: {validation_result}")
