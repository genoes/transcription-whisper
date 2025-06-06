import whisper
from moviepy import VideoFileClip, AudioFileClip


# Step 1: Extract audio from MP4 (Whisper needs WAV/MP3)
mp4_file = input('\n'"Enter video file path: ").strip().replace("\\ ", " ")
audio_file = "extracted_audio.wav"

# Convert MP4 to WAV
video = AudioFileClip(mp4_file)
video.write_audiofile(audio_file, codec='pcm_s16le')  # WAV format

# Step 2: Transcribe with Whisper
model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy
result = model.transcribe(audio_file)

# Transcribe with language detection and precision settings
result = model.transcribe(
    audio_file,
    language="en",         # Force English (omit for auto-detect)
    fp16=False,            # Disable if no GPU
    verbose=True           # Show progress
)

# Print results
print("Transcription:")
print(result["text"])

# Save to file with timestamps
with open("transcription.txt", "w") as f:
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Convert seconds to minutes:seconds format
        start_min, start_sec = divmod(start_time, 60)
        end_min, end_sec = divmod(end_time, 60)
        
        timestamp = f"[{int(start_min):02d}:{start_sec:06.3f}-{int(end_min):02d}:{end_sec:06.3f}]"
        f.write(f"{timestamp} {text}\n\n")