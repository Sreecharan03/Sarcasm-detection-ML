"""
Model Loader for Sarcasm Detection System
Loads trained hybrid model and provides prediction interface
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class ModelLoader:
    """
    Production model loader for sarcasm detection
    Handles hybrid model (TF-IDF + contextual features)
    """
    
    def __init__(self, model_dir='saved_models'):
        """Initialize model loader with saved model components"""
        self.model_dir = Path(model_dir)
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.config = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_loaded = False 
        
        # Load all model components
        self._load_components()
    
    def _load_components(self):
        """Load all saved model components"""
        try:
            # Load production configuration
            config_path = self.model_dir / 'production_config.json'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Load model components
            model_path = self.model_dir / 'production_model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)

            tfidf_path = self.model_dir / 'tfidf_vectorizer.pkl'
            if not tfidf_path.exists():
                raise FileNotFoundError(f"TF-IDF vectorizer file not found: {tfidf_path}")
            self.tfidf_vectorizer = joblib.load(tfidf_path)

            scaler_path = self.model_dir / 'feature_scaler.pkl'
            if not scaler_path.exists():
                raise FileNotFoundError(f"Feature scaler file not found: {scaler_path}")
            self.feature_scaler = joblib.load(scaler_path)
            
            # Use 'self.config' only after confirming it's loaded
            if self.config and self.model and self.tfidf_vectorizer and self.feature_scaler:
                print(f"✅ Loaded {self.config['model_type']} model (F1: {self.config['f1_score']:.3f})")
                self.is_loaded = True
            else:
                raise RuntimeError("One or more essential model components (model, config, tfidf, scaler) failed to load properly.")
            
        except Exception as e:
            print(f"❌ Failed to load model components: {e}")
            self.is_loaded = False
    
    def _extract_contextual_features(self, text):
        """Extract contextual features from text (same as training)"""
        features = {}
        
        # Sentiment features
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_compound'] = sentiment_scores['compound']
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        features['sentiment_neutral'] = sentiment_scores['neu']
        
        blob = TextBlob(text)
        # Directly access polarity and subjectivity from blob.sentiment
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Incongruity features
        positive_words = ['great', 'wonderful', 'amazing', 'perfect', 'brilliant', 'fantastic', 'excellent']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'stupid', 'ridiculous']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        features['positive_words'] = pos_count
        features['negative_words'] = neg_count
        features['sentiment_incongruity'] = 1 if (pos_count > 0 and neg_count > 0) else 0
        
        # Hyperbole features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        
        intensifiers = ['so', 'very', 'extremely', 'absolutely', 'totally', 'completely', 'really']
        features['intensifier_count'] = sum(1 for word in intensifiers if word in text_lower)
        
        # Lexical features
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['quote_count'] = text.count('"') + text.count("'")
        features['ellipsis_count'] = text.count('...')
        
        return features
    
    def _prepare_features(self, texts):
        """Prepare hybrid features for prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model components or configuration are not loaded. Cannot prepare features.")

        if isinstance(texts, str):
            texts = [texts]
        
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not loaded.")
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # Contextual features
        contextual_data = []
        for text in texts:
            features = self._extract_contextual_features(text)
            contextual_data.append(features)
        
        contextual_df = pd.DataFrame(contextual_data)
        
        # Ensure correct column order, safely access self.config
        if self.config is None:
            raise RuntimeError("Configuration is None, cannot access 'contextual_feature_names'.")
        expected_columns = self.config['contextual_feature_names']
        contextual_df = contextual_df.reindex(columns=expected_columns, fill_value=0)
        
        # Scale contextual features
        if self.feature_scaler is None:
            raise RuntimeError("Feature scaler not loaded.")
        contextual_scaled = self.feature_scaler.transform(contextual_df)
        
        # Combine features
        hybrid_features = hstack([tfidf_features, contextual_scaled])
        
        return hybrid_features
    
    def predict(self, text):
        """
        Predict sarcasm for a single text
        Returns dict with prediction details
        """
        if not self.is_loaded or self.model is None:
            return {
                'text': text,
                'error': "Model components are not loaded or model is missing.",
                'sarcasm_probability': 0.0,
                'label': 'error',
                'confidence': 0.0
            }
        try:
            features = self._prepare_features([text])
            prediction_proba = self.model.predict_proba(features)[0]
            prediction = self.model.predict(features)[0]
            
            return {
                'text': text,
                'sarcasm_probability': float(prediction_proba[1]),
                'label': 'sarcasm' if prediction == 1 else 'not_sarcasm',
                'confidence': float(max(prediction_proba)),
                'model_type': self.config['model_type']
            }
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'sarcasm_probability': 0.0,
                'label': 'error',
                'confidence': 0.0
            }
    
    def predict_batch(self, texts):
        """
        Predict sarcasm for multiple texts
        Returns list of prediction dictionaries
        """
        if not self.is_loaded or self.model is None:
            return [{'text': text, 'error': "Model components are not loaded or model is missing."} for text in texts]
        try:
            features = self._prepare_features(texts)
            predictions_proba = self.model.predict_proba(features)
            predictions = self.model.predict(features)
            
            results = []
            for i, text in enumerate(texts):
                result = {
                    'text': text,
                    'sarcasm_probability': float(predictions_proba[i][1]),
                    'label': 'sarcasm' if predictions[i] == 1 else 'not_sarcasm',
                    'confidence': float(max(predictions_proba[i])),
                    'model_type': self.config['model_type']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{'text': text, 'error': str(e)} for text in texts]
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_loaded or self.config is None:
            return {
                'model_type': 'Unknown',
                'f1_score': 0.0,
                'feature_counts': 0,
                'target_achieved': False,
                'error': 'Model not loaded or configuration is missing'
            }
        return {
            'model_type': self.config['model_type'],
            'f1_score': self.config['f1_score'],
            'feature_counts': self.config['feature_counts'],
            'target_achieved': self.config['target_achieved']
        }


def main():
    """
    Main function for testing the model loader
    Run this file directly to test model loading and predictions
    """
    print("🤖 Testing Sarcasm Detection Model Loader")
    print("=" * 45)
    
    try:
        # Load the model
        print("📂 Loading model...")
        loader = ModelLoader()
        
        if not loader.is_loaded:
            print(f"❌ Model loading failed. Please ensure 'saved_models/' directory and its contents are correct.")
            return # Exit if model loading failed
            
        # Show model info
        info = loader.get_model_info()
        print(f"✅ Model loaded successfully!")
        print(f"   Type: {info['model_type']}")
        print(f"   F1-Score: {info['f1_score']:.3f}")
        print(f"   Features: {info['feature_counts']}")
        
        # Test single predictions
        print(f"\n🧪 Testing Single Predictions:")
        print("-" * 35)
        
        test_texts = [
            "What a wonderful day to be stuck in traffic!",  # Sarcastic
            "I really enjoyed the movie last night.",       # Not sarcastic  
            "Oh great, another meeting about meetings.",     # Sarcastic
            "Thank you for your help with the project.",    # Not sarcastic
            "Perfect timing for the server to crash!",      # Sarcastic
            "The weather is nice today."                    # Not sarcastic
        ]
        
        for text in test_texts:
            result = loader.predict(text)
            label = result['label']
            prob = result['sarcasm_probability']
            confidence = result['confidence']
            
            # Format output with emoji
            emoji = "🙄" if label == 'sarcasm' else "😊"
            print(f"{emoji} [{label.upper()}] {prob:.3f} | {text}")
        
        # Test batch predictions
        print(f"\n📊 Testing Batch Predictions:")
        print("-" * 30)
        
        batch_texts = [
            "Love getting up early on weekends!",
            "This is genuinely helpful information.",
            "Another brilliant idea from management."
        ]
        
        batch_results = loader.predict_batch(batch_texts)
        sarcastic_count = sum(1 for r in batch_results if r['label'] == 'sarcasm')
        
        print(f"Processed {len(batch_results)} texts:")
        print(f"   Sarcastic: {sarcastic_count}")
        print(f"   Not Sarcastic: {len(batch_results) - sarcastic_count}")
        
        for result in batch_results:
            label = result['label']
            prob = result['sarcasm_probability']
            print(f"   {label}: {prob:.3f} - {result['text']}")
        
        print(f"\n🎉 Model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        print(f"💡 Ensure model files are correctly placed in the 'saved_models/' directory and are valid.")

if __name__ == "__main__":
    main()

def load_model():
    """Loads and returns the sarcasm detection model."""
    return ModelLoader()

def process_text(text, model):
    """Processes text and returns the model's prediction."""
    result = model.predict(text)
    return result['label']
