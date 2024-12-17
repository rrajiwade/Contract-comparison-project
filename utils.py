import os
import logging
from pdf2image import convert_from_path
import pytesseract
from joblib import load
import numpy as np

def extract_text_from_pdf(pdf_path):
    try:
        logging.info(f"Starting OCR for PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        text = ''
        for idx, image in enumerate(images):
            logging.info(f"Processing page {idx + 1} of {pdf_path}")
            text += pytesseract.image_to_string(image)
        logging.info(f"OCR completed for PDF: {pdf_path}")
        return text
    except Exception as e:
        logging.error(f"Error during PDF to image conversion or OCR for PDF '{pdf_path}': {str(e)}")
        return ""
    pass

def classify_document(pdf_path):
    try:
        logging.info(f"Classifying document: {pdf_path}")
        model_dir = os.path.join(os.getcwd(), 'backend', 'model')
        classifier_path = os.path.join(model_dir, 'document_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        vocab_path = os.path.join(model_dir, 'tfidf_vocab.joblib')

        classifier = load(classifier_path)
        vectorizer = load(vectorizer_path)
        vocabulary = load(vocab_path)

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "Insufficient text extracted", 0, []

        X_new = vectorizer.transform([text])

        predicted_label = classifier.predict(X_new)[0]
        match_percentage = classifier.predict_proba(X_new)[0].max() * 100

        # Get keywords influencing the classification
        top_n = 5  # Number of top keywords to show
        feature_array = np.array(vocabulary)
        tfidf_scores = X_new.toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        top_keywords = feature_array[top_indices].tolist()  # Convert ndarray to list

        logging.info(f"Prediction: {predicted_label}, Match: {match_percentage:.2f}%")
        logging.info(f"Top keywords: {top_keywords}")

        return predicted_label, match_percentage, top_keywords

    except Exception as e:
        logging.error(f"Error classifying document '{pdf_path}': {str(e)}")
        return "Unknown", 0, []
