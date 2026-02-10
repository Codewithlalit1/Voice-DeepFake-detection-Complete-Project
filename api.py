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
import gc

# Matplotlib configuration - MUST be before importing pyplot
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp/matplotlib', exist_ok=True)

import matplotlib.pyplot as plt
plt.ioff()

app = FastAPI(
    title="Voice Deepfake Detection API",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def create_spectrogram(audio_data, sr):
    """
    Generates a memory-efficient spectrogram visualization
    """
    fig = None
    try:
        # Limit audio length to 30 seconds
        max_duration = 30
        max_samples = sr * max_duration
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
            print(f"⚠️ Audio truncated to {max_duration}s for memory efficiency")
        
        # Create smaller figure
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        
        # Convert amplitude to Decibels for visualization (ORIGINAL METHOD)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        # Draw the image (ORIGINAL STYLE)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Frequency Spectrogram')
        plt.tight_layout()
        
        # Save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Read buffer
        image_bytes = buf.read()
        buf.close()
        
        # Clean up matplotlib
        plt.close(fig)
        plt.clf()
        plt.close('all')
        
        # Clean up arrays
        del D
        gc.collect()
        
        # Validate
        if len(image_bytes) == 0:
            print("❌ ERROR: Image buffer is empty!")
            return None
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        del image_bytes
        
        print(f"✅ Spectrogram generated: {len(image_base64)} chars")
        return image_base64
        
    except Exception as e:
        print(f"❌ Spectrogram generation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if fig is not None:
            plt.close(fig)
        plt.close('all')
        gc.collect()
        
        return None

def extract_features(audio_data, sample_rate):
    """
    Extract MFCC features from audio
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        # Clean up
        del mfccs
        gc.collect()
        
        return mfccs_processed
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Voice Deepfake Detection API",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict if audio is AI-generated or real human voice
    """
    temp_filename = f"/tmp/temp_{file.filename}"
    
    try:
        print(f"\n{'='*60}")
        print(f"📁 Processing file: {file.filename}")
        
        # Check file size (limit to 10MB)
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return {"error": "File too large. Maximum size is 10MB."}
        
        print(f"📦 File size: {file_size / 1024:.2f} KB")
        
        # Save to temp file
        with open(temp_filename, "wb") as f:
            f.write(contents)
        
        del contents
        gc.collect()
        
        # Load audio with optimizations
        print("🎵 Loading audio...")
        audio, sample_rate = librosa.load(
            temp_filename,
            sr=22050,  # Standard sample rate
            duration=30,  # Load max 30 seconds
            res_type='kaiser_fast'
        )
        
        print(f"✅ Audio loaded: {len(audio)} samples at {sample_rate}Hz")
        
        # Generate spectrogram
        print("🎨 Generating spectrogram...")
        spectrogram_image = create_spectrogram(audio.copy(), sample_rate)
        
        # Extract features for prediction
        print("🔍 Extracting features...")
        features = extract_features(audio, sample_rate)
        
        # Clean up audio data immediately
        del audio
        gc.collect()
        
        # Delete temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print("🗑️  Temp file deleted")

        if features is None:
            return {"error": "Could not extract audio features"}
        
        if model is None:
            return {"error": "Model not loaded"}
        
        # Make prediction
        print("🤖 Running AI prediction...")
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        probability = model.predict_proba(features_reshaped)[0]
        
        result = "AI_FAKE" if prediction == 1 else "REAL_HUMAN"
        confidence = float(probability[prediction] * 100)
        dominant_index = int(np.argmax(np.abs(features)))
        
        # Clean up
        del features, features_reshaped
        gc.collect()
        
        print(f"✅ Prediction: {result} ({confidence:.2f}%)")
        print(f"{'='*60}\n")
        
        return {
            "result": result,
            "confidence": f"{confidence:.2f}%",
            "spectrogram": spectrogram_image if spectrogram_image else "",
            "dominant_feature_index": dominant_index
        }

    except Exception as e:
        print(f"❌ ERROR in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        
        gc.collect()
        
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run: & "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn api:app
# .\venv\Scripts\Activate.ps1
