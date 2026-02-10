import requests

# CHANGE THIS to your actual audio file path
audio_file_path = r"C:\Users\ASUS\Downloads\Priyanshu_voice2.ogg"

url = "https://voice-deepfake-detection-backend.onrender.com/predict"
#https://voice-deepfake-detection-backend.onrender.com/predict
#http://localhost:8000/predict

try:
    with open(audio_file_path, 'rb') as f:
        files = {'file': ('audio.wav', f, 'audio/wav')}
        response = requests.post(url, files=files, timeout=60)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS!")
        print(f"Result: {data.get('result')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Has Spectrogram: {bool(data.get('spectrogram'))}")
        print(f"Spectrogram Length: {len(data.get('spectrogram', ''))} chars")
    else:
        print(f"\n❌ Error {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")