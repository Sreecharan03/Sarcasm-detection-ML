"""
Text Processor for Sarcasm Detection System
Handles text cleaning, validation, and preprocessing utilities
"""

import re
import string
import pandas as pd
from typing import Union, List, Dict


class TextProcessor:
    """
    Text processing utilities for sarcasm detection
    Handles cleaning, validation, and format standardization
    """
    
    def __init__(self):
        """Initialize text processor with cleaning rules"""
        # Contractions mapping for expansion
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "'s": " is"
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for sarcasm detection (preserves sarcasm patterns)
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove or replace URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
        
        # Handle email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Expand contractions (important for sarcasm detection)
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Handle excessive punctuation (preserve some for sarcasm)
        text = re.sub(r'[!]{3,}', '!!', text)  # Max 2 exclamation marks
        text = re.sub(r'[?]{3,}', '??', text)  # Max 2 question marks
        text = re.sub(r'[.]{4,}', '...', text)  # Standardize ellipsis
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation (but keep internal)
        text = text.strip(string.punctuation)
        
        return text
    
    def validate_text(self, text: str) -> Dict[str, Union[bool, str]]:
        """
        Validate text input for processing
        
        Args:
            text: Text to validate
            
        Returns:
            Dict with validation result and message
        """
        if not text or pd.isna(text):
            return {"valid": False, "message": "Text is empty or null"}
        
        # Convert to string
        text_str = str(text).strip()
        
        # Check minimum length
        if len(text_str) < 3:
            return {"valid": False, "message": "Text too short (minimum 3 characters)"}
        
        # Check maximum length
        if len(text_str) > 1000:
            return {"valid": False, "message": "Text too long (maximum 1000 characters)"}
        
        # Check if text is mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in text_str) / len(text_str)
        if alpha_ratio < 0.3:
            return {"valid": False, "message": "Text contains too few alphabetic characters"}
        
        # Check for suspicious content
        suspicious_patterns = [
            r'^[^a-zA-Z]*$',  # Only non-alphabetic characters
            r'^(\w)\1{10,}',  # Repeated characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text_str):
                return {"valid": False, "message": "Text contains suspicious patterns"}
        
        return {"valid": True, "message": "Text is valid"}
    
    def standardize_format(self, text: str) -> str:
        """
        Standardize text format for consistent processing
        
        Args:
            text: Text to standardize
            
        Returns:
            Standardized text
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Ensure single sentence format (remove multiple sentences for consistency)
        # Split by sentence endings and take first meaningful sentence
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            # Take the first non-empty sentence
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 3:
                    text = sentence
                    break
        
        # Add period if missing (helps with consistency)
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def process_text(self, text: str, validate: bool = True, standardize: bool = True) -> Dict[str, Union[str, bool, Dict]]:
        """
        Complete text processing pipeline
        
        Args:
            text: Raw text input
            validate: Whether to validate input
            standardize: Whether to standardize format
            
        Returns:
            Dict with processed text and metadata
        """
        result = {
            "original_text": text,
            "processed_text": "",
            "valid": True,
            "validation_message": "",
            "processed": False
        }
        
        try:
            # Validation step
            if validate:
                validation = self.validate_text(text)
                result["valid"] = validation["valid"]
                result["validation_message"] = validation["message"]
                
                if not validation["valid"]:
                    return result
            
            # Cleaning step
            processed = self.clean_text(text)
            
            # Standardization step
            if standardize:
                processed = self.standardize_format(processed)
            
            result["processed_text"] = processed
            result["processed"] = True
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["validation_message"] = f"Processing error: {str(e)}"
            return result
    
    def process_batch(self, texts: List[str], validate: bool = True, standardize: bool = True) -> List[Dict]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of text strings
            validate: Whether to validate inputs
            standardize: Whether to standardize format
            
        Returns:
            List of processing results
        """
        results = []
        for text in texts:
            result = self.process_text(text, validate=validate, standardize=standardize)
            results.append(result)
        
        return results
    
    def get_text_stats(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get basic text statistics
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with text statistics
        """
        if not text:
            return {"char_count": 0, "word_count": 0, "sentence_count": 0, "avg_word_length": 0}
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "punctuation_count": sum(1 for c in text if c in string.punctuation)
        }


def main():
    """
    Main function for testing text processor
    Run this file directly to test text processing functions
    """
    print("📝 Testing Text Processor")
    print("=" * 30)
    
    # Initialize processor
    processor = TextProcessor()
    
    # Test texts
    test_texts = [
        "What a WONDERFUL day to be stuck in traffic!!!",  # Messy formatting
        "can't believe this happened... won't happen again",  # Contractions
        "Check out this link: https://example.com awesome!",  # URLs
        "   extra   whitespace   everywhere   ",  # Whitespace
        "abc",  # Too short
        "🙄😤🤬",  # Only emojis/symbols
        "This is a normal sentence.",  # Clean text
        "",  # Empty text
        "A" * 1001  # Too long
    ]
    
    print("🧪 Testing Text Processing:")
    print("-" * 35)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Testing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        result = processor.process_text(text)
        
        if result["valid"]:
            print(f"   ✅ Valid → '{result['processed_text']}'")
            
            # Get text stats
            stats = processor.get_text_stats(result['processed_text'])
            print(f"   📊 Stats: {stats['word_count']} words, {stats['char_count']} chars")
            
        else:
            print(f"   ❌ Invalid: {result['validation_message']}")
    
    # Test batch processing
    print(f"\n📊 Testing Batch Processing:")
    print("-" * 25)
    
    batch_texts = [
        "Great weather for outdoor activities!",
        "Another meeting that could've been an email...",
        "Thanks for the help!"
    ]
    
    batch_results = processor.process_batch(batch_texts)
    valid_count = sum(1 for r in batch_results if r["valid"])
    
    print(f"Processed {len(batch_results)} texts:")
    print(f"   Valid: {valid_count}")
    print(f"   Invalid: {len(batch_results) - valid_count}")
    
    for result in batch_results:
        if result["valid"]:
            original = result["original_text"][:30]
            processed = result["processed_text"][:30]
            print(f"   '{original}' → '{processed}'")
    
    print(f"\n🎉 Text processor testing completed!")


if __name__ == "__main__":
    main()