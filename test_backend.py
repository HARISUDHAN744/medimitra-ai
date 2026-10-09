import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app
from app.config import SAMPLES_DIR

def run_tests():
    client = TestClient(app)
    
    print("--- 1. Testing Health Endpoint ---")
    res = client.get("/health")
    print(f"Status: {res.status_code}, Response: {res.json()}")
    assert res.status_code == 200, "Health check failed"

    print("\n--- 2. Testing Samples List Endpoint ---")
    res = client.get("/samples")
    print(f"Status: {res.status_code}")
    samples_data = res.json()
    print(f"Found {len(samples_data.get('samples', []))} sample documents.")
    assert len(samples_data.get('samples', [])) >= 3, "Expected at least 3 samples"

    print("\n--- 3. Testing Upload Endpoint (Sample Prescription 1) ---")
    sample_file_path = SAMPLES_DIR / "sample_prescription_1.png"
    assert sample_file_path.exists(), f"Sample image {sample_file_path} does not exist"
    
    with open(sample_file_path, "rb") as f:
        file_bytes = f.read()
    
    res = client.post(
        "/upload",
        files={"file": ("sample_prescription_1.png", file_bytes, "image/png")}
    )
    print(f"Upload Status: {res.status_code}")
    data = res.json()
    print(f"Success: {data.get('success')}")
    print(f"File Hash: {data.get('file_hash')}")
    print(f"Model Used: {data.get('model_used')}")
    print(f"Patient Name: {data.get('data', {}).get('patient_name')}")
    assert res.status_code == 200, "First upload failed"

    print("\n--- 4. Testing Arbitrary / Custom Image Upload ---")
    from PIL import Image
    import io
    custom_img = Image.new("RGB", (600, 800), color=(240, 248, 255))
    buf = io.BytesIO()
    custom_img.save(buf, format="PNG")
    custom_bytes = buf.getvalue()
    
    res_custom = client.post(
        "/upload",
        files={"file": ("Dr_Priya_Cardio_Consultation.png", custom_bytes, "image/png")}
    )
    data_custom = res_custom.json()
    print(f"Custom Image Upload Status: {res_custom.status_code}")
    print(f"Extracted Document Type: {data_custom.get('data', {}).get('document_type')}")
    assert res_custom.status_code == 200, "Custom image upload failed"

    print("\n--- 5. Testing Universal Audio Transcription Endpoint (/transcribe) ---")
    fake_audio_bytes = b"ID3\x03\x00\x00\x00\x00\x00#TSSE\x00\x00\x00\x0f\x00\x00\x01\xff\xfeL\x00a\x00v\x00f\x005\x008\x00.\x007\x006\x00.\x001\x000\x000\x00" + b"\x00" * 200
    res_transcribe = client.post(
        "/transcribe",
        files={"file": ("doctor_note_audio.mp3", fake_audio_bytes, "audio/mpeg")},
        data={"language": "en", "extract_medical": "true"}
    )
    print(f"Transcribe Status: {res_transcribe.status_code}")
    data_transcribe = res_transcribe.json()
    print(f"Success: {data_transcribe.get('success')}")
    print(f"Provider: {data_transcribe.get('provider')}")
    print(f"Extracted Medical: {bool(data_transcribe.get('extracted_medical'))}")
    assert res_transcribe.status_code == 200, "Transcribe endpoint failed"

    print("\n--- 6. Testing JSON Body Text Extraction (/transcribe) ---")
    res_text = client.post(
        "/transcribe",
        json={
            "text": "Patient has severe hypertension and type 2 diabetes. Prescribed Metformin 500mg twice daily and Amlodipine 5mg once daily. HbA1c test recommended.",
            "language": "en",
            "extract_medical": True
        }
    )
    print(f"Text Extraction Status: {res_text.status_code}")
    data_text = res_text.json()
    print(f"Success: {data_text.get('success')}")
    print(f"Extracted Diagnoses: {len(data_text.get('extracted_medical', {}).get('diagnoses', []))}")
    print(f"Extracted Medications: {len(data_text.get('extracted_medical', {}).get('medications', []))}")
    assert res_text.status_code == 200, "Text extraction failed"
    assert data_text.get("extracted_medical") is not None, "Expected extracted medical data"

    print("\n==========================================")
    print("[SUCCESS] ALL BACKEND & AUDIO TESTS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
