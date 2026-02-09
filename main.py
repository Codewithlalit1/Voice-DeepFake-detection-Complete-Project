import os
import numpy as np
import librosa
import joblib
import argparse

# --- CONFIGURATION ---
MODEL_FILE = 'voice_detector_model.pkl'

def extract_features(file_path):
    """
    Same logic as training: Load audio -> Extract MFCCs -> Average them
    """
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_voice(file_path):
    # 1. Check if model exists
    if not os.path.exists(MODEL_FILE):
        print("Error: Model not found! Run train.py first.")
        return

    # 2. Load the trained model
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    
    # 3. Process the audio file
    print(f"Analyzing {file_path}...")
    features = extract_features(file_path)
    
    if features is None:
        print("Could not read file.")
        return

    # 4. Make Prediction
    # Reshape features because model expects a list of files, not just one
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # 5. Interpret Results
    result = "FAKE (AI)" if prediction == 1 else "REAL (Human)"
    confidence = probabilities[prediction] * 100
    
    print("\n" + "="*30)
    print(f"   RESULT: {result}")
    print(f"   Confidence: {confidence:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    # Allow user to type file path in terminal
    file_path = input("Enter the path to the audio file (e.g., test.wav): ").strip().strip('"')
    
    if os.path.exists(file_path):
        predict_voice(file_path)
    else:
        print("Error: File not found.")