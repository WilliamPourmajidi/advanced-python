#!/usr/bin/env python3
"""
meeting_transcription_ide.py

Extracts audio from an MP4 video, transcribes it to text, and summarizes
the transcript into structured meeting minutes—all with hardcoded
parameters for easy execution from your IDE (PyCharm, VS Code, etc.).

This style is often called “script mode” or running via an IDE Run Configuration:
you define configuration variables at the top instead of passing CLI arguments.
"""

import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# === Configuration (edit these, then simply run the script in your IDE) ===
video_path         = "data_samples\meeting.mp4"       # Path to your meeting video
output_dir         = "output"            # Where outputs will be saved
use_local_whisper  = True               # True to use local Whisper; False to use OpenAI API
whisper_model      = "base"              # Whisper model: tiny, base, small, medium, large
openai_key         = "YOUR_OPENAI_KEY"   # Your OpenAI API key (required if not using local Whisper)
summarize_minutes  = True                # Whether to generate meeting minutes

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)
audio_path     = os.path.join(output_dir, "meeting.wav")
transcript_path= os.path.join(output_dir, "transcript.txt")
minutes_path   = os.path.join(output_dir, "minutes.txt")

def extract_audio(video_src, audio_dst):
    """
    Extracts audio from video and saves as WAV with 16 kHz sampling rate and 16-bit PCM.
    """
    print(f"[1/3] Extracting audio from {video_src} …")
    clip = VideoFileClip(video_src)
    clip.audio.write_audiofile(audio_dst, fps=16000, nbytes=2, codec='pcm_s16le')
    print("    → Audio saved to:", audio_dst)

def transcribe_local(audio_src, model_name="base"):
    """
    Uses a local Whisper model to transcribe audio to text.
    """
    import whisper
    print(f"[2/3] Loading local Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)
    print("    Transcribing audio …")
    result = model.transcribe(audio_src)
    return result["text"]

def transcribe_api(audio_src, api_key):
    """
    Uses OpenAI Whisper API to transcribe audio to text.
    """
    import openai
    openai.api_key = api_key
    print("[2/3] Transcribing via OpenAI Whisper API …")
    with open(audio_src, "rb") as f:
        resp = openai.Audio.transcribe("whisper-1", f)
    return resp["text"]

def summarize_api(transcript, api_key, model="gpt-3.5-turbo"):
    """
    Sends transcript to OpenAI chat endpoint to generate structured meeting minutes.
    """
    import openai
    openai.api_key = api_key
    prompt = f"""
You are an assistant that turns raw meeting transcripts into clear, concise minutes.
Please produce:
1. Meeting title
2. Date & attendees
3. 3–5 bullet-point action items
4. Decisions made
5. Brief summary of each agenda topic

Transcript:
{transcript}
"""
    print("[3/3] Generating meeting minutes …")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# === Main execution flow ===
if __name__ == "__main__":
    # 1) Extract audio
    extract_audio(video_path, audio_path)

    # 2) Transcribe
    if use_local_whisper:
        transcript = transcribe_local(audio_path, whisper_model)
    else:
        if not openai_key:
            raise ValueError("Set `openai_key` or enable `use_local_whisper` in configuration.")
        transcript = transcribe_api(audio_path, openai_key)

    # Save transcript
    with open(transcript_path, "w") as f:
        f.write(transcript)
    print("    → Transcript saved to:", transcript_path)

    # 3) Summarize if requested
    if summarize_minutes:
        if not openai_key and not use_local_whisper:
            raise ValueError("Summarization requires `openai_key` unless `use_local_whisper` is True.")
        minutes = summarize_api(transcript, openai_key)
        with open(minutes_path, "w") as f:
            f.write(minutes)
        print("    → Meeting minutes saved to:", minutes_path)

print("\nDone! simply hit Run in your IDE to re-execute.")
