import whisper

# Step 1: Get WAV file input
wav_file = input('\n'"Enter WAV audio file path: ").strip().replace("\\ ", " ")

# Step 2: Transcribe with Whisper
model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy

# Transcribe with language detection and precision settings
result = model.transcribe(
    wav_file,
    language="en",         # Force English (omit for auto-detect)
    fp16=False,            # Disable if no GPU
    verbose=True           # Show progress
)

# Print results
print("\nTranscription:")
print(result["text"])

# Save to file with timestamps
with open("audio-transcription.txt", "w") as f:
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Convert seconds to minutes:seconds format
        start_min, start_sec = divmod(start_time, 60)
        end_min, end_sec = divmod(end_time, 60)
        
        timestamp = f"[{int(start_min):02d}:{start_sec:06.3f}-{int(end_min):02d}:{end_sec:06.3f}]"
        f.write(f"{timestamp} {text}\n\n")

print("\nTranscription saved to 'transcription.txt'")