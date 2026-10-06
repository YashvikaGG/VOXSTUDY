from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")

segments, info = model.transcribe("sample.mp3")

print("\nTranscription:")

for segment in segments:
    print(segment.text)