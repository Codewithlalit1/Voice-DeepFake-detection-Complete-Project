import librosa
import numpy as np

def extract_features(file_path):
    # Loads the audio file
    # sr=None preserves the original sampling rate
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # 1. MFCC (Mel-Frequency Cepstral Coefficients)
        # This acts like a "fingerprint" of the vocal tract (throat shape)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
        
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None