from src.model_loader import load_model, process_text
from src.llm_validator import validate_sarcasm

def main():
    """
    Main function to run the sarcasm detection application.
    """
    # Load the ML model
    model = load_model()

    # Get user input
    text = input("Enter a sentence to check for sarcasm: ")

    # Process the text and get the ML model's prediction
    prediction = process_text(text, model)
    print(f"ML Model Prediction: {prediction}")

    # Validate the prediction with the Gemini model
    validation = validate_sarcasm(text, prediction)
    print(f"Final Output: {validation}")

if __name__ == "__main__":
    main()

