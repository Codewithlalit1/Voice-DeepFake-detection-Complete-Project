from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import librosa.display
import joblib
import io
import os
import shutil
import base64

# ====== MATPLOTLIB CONFIGURATION FOR DEPLOYMENT ======
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

# Set config directory to tmp (writable on most platforms)
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp/matplotlib', exist_ok=True)

# Disable font manager warnings and cache
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
# ====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "voice_detector_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    print("Model not found. Train it first!")

def create_spectrogram(audio_data, sr):
    """
    Generates a visual 'Heatmap' of the audio frequency.
    """
    try:
        fig = plt.figure(figsize=(10, 4))
        
        # Convert amplitude to Decibels for visualization
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        # Draw the image
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Frequency Spectrogram')
        plt.tight_layout()
        
        # Save to a memory buffer (not a file)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Close the specific figure
        plt.clf()  # Clear the current figure
        
        # Encode as Base64 string to send to React
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64
    except Exception as e:
        print(f"Spectrogram generation error: {e}")
        return None

def extract_features(audio_data, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except:
        return None

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    temp_filename = f"/tmp/temp_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load Audio
        audio, sample_rate = librosa.load(temp_filename, res_type='kaiser_fast')
        
        # 1. Generate the Spectrogram Image
        print("🎨 Generating spectrogram...")
        spectrogram_image = create_spectrogram(audio, sample_rate)
        
        if spectrogram_image is None:
            print("❌ Spectrogram generation failed!")
        else:
            print(f"✅ Spectrogram generated: {len(spectrogram_image)} chars")

        # 2. Extract Features for AI
        features = extract_features(audio, sample_rate)
        
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if features is None:
            return {"error": "Could not process audio"}
        
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        probability = model.predict_proba(features_reshaped)[0]
        
        result = "AI_FAKE" if prediction == 1 else "REAL_HUMAN"
        confidence = float(probability[prediction] * 100)
        
        # Find dominant feature
        dominant_index = np.argmax(np.abs(features))
        
        return {
            "result": result,
            "confidence": f"{confidence:.2f}%",
            "spectrogram": spectrogram_image if spectrogram_image else "",  # Send empty string if None
            "dominant_feature_index": int(dominant_index)
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {"error": str(e)}

# Run: & "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn api:app
# .\venv\Scripts\Activate.ps1
