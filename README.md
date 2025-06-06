# Video/Audio Transcription Tool

A Python application that extracts audio from video files and transcribes it using OpenAI's Whisper model, with timestamped output.

## Features

- Extracts audio from MP4 files
- Transcribes audio using Whisper speech recognition
- Generates timestamps for each transcribed segment
- Supports multiple Whisper model sizes (base, small, medium, large)
- Outputs transcription with time markers to a text file

## Requirements

- Python 3.7+
- FFmpeg (must be installed and added to system PATH)
- GPU recommended for faster processing (but works on CPU)

## Installation

1. Clone this repository or download the script
2. Install required packages:

```bash
pip3 install -r requirements.txt
