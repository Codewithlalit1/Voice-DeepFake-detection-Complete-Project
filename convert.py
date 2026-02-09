import os
from pydub import AudioSegment

# Define your paths
REAL_FOLDER = "data/real/"
FAKE_FOLDER = "data/fake/"

def convert_to_wav(folder_path):
    print(f"Processing folder: {folder_path}")
    files = os.listdir(folder_path)
    
    for filename in files:
        # Check for non-wav audio files
        if filename.lower().endswith(('.mp3', '.mpeg', '.m4a', '.ogg')):
            full_path = os.path.join(folder_path, filename)
            file_name_no_ext = os.path.splitext(filename)[0]
            wav_path = os.path.join(folder_path, file_name_no_ext + ".wav")
            
            print(f"Converting: {filename} -> {file_name_no_ext}.wav")
            
            try:
                # Load the file
                audio = AudioSegment.from_file(full_path)
                # Export as WAV (Standard 16-bit PCM)
                audio.export(wav_path, format="wav")
                
                # OPTIONAL: Delete the old file to clean up
                os.remove(full_path) 
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Run conversion on both folders
convert_to_wav(REAL_FOLDER)
convert_to_wav(FAKE_FOLDER)
print("Conversion Complete! You are ready to train.")