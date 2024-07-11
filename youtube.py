import torch
#from transformers import BartTokenizer, AutoTokenizer
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    video_id = re.search(r"(?<=v=)[^&#]+", url)
    if video_id:
        return video_id.group(0)
    video_id = re.search(r"(?<=be/)[^&#]+", url)
    return video_id.group(0) if video_id else None

# Function to get transcript from YouTube video
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    if not transcript:
        st.write(f"Vacio")
    transcript_text = " ".join([item['text'] for item in transcript])
    return transcript_text

# Define the summarization function
def summarize_from_youtube(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL"
    
    try:
        transcript = get_transcript(video_id)
        inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=1024, truncation=True)
    except Exception as e:
        return f"Error processing transcript: {str(e)}"
    
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
st.title("YouTube Video Summarizer")
st.write("Enter a YouTube URL to get the transcript and summary of the video.")

url = st.text_input("YouTube URL")

if st.button("Summarize"):
    if url:
        summary = summarize_from_youtube(url)
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter a YouTube URL")

