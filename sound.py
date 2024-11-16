import nest_asyncio
import pyaudio
import torch
import torchaudio
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os
import warnings, torch, transformers, deepl
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args, **kwargs: None

# Device and Tensor Type
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float16

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

key = "0b32b5cd-952d-5482-807c-84a979d8fd7d:fx" 
translator = deepl.Translator(key)

nest_asyncio.apply()  # Apply the nest_asyncio patch

# Load Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model.to(device)

# DeepL API configuration
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_KEY = key

# Real-time audio capture settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Function to call DeepL API
def translate_with_deepl(text, target_lang="EN"):
    data = {
        "auth_key": key,
        "text": text,
        "target_lang": target_lang
    }
    response = requests.post(DEEPL_API_URL, data=data)
    result = response.json()
    print("DeepL Response:", result)  # Debug print
    return result["translations"][0]["text"]

# Function to create SRT file
def create_srt(transcripts, filename="transcript.srt"):
    with open(filename, "w") as f:
        for i, (start_time, end_time, text) in enumerate(transcripts, 1):
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file saved to {os.path.abspath(filename)}")  # Print the file path

# Capture audio from Virtual Audio Cable
def capture_audio(queue):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=1)  # Index may vary
    print("* Recording audio...")

    while True:
        data = stream.read(CHUNK)
        queue.put(data)

# Convert time to SRT format
def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

# Asynchronous transcription and translation
async def process_chunk(chunk, start_time):
    audio_input = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
    features = whisper_processor(audio_input, sampling_rate=RATE, return_tensors="pt").to(device)
    transcription = whisper_model.generate(**features)
    transcribed_text = whisper_processor.batch_decode(transcription, skip_special_tokens=True)[0]
    print("Transcribed Text:", transcribed_text)  # Debug print
    
    translated_text = translate_with_deepl(transcribed_text, target_lang="EN")
    print("Translated Chunk:", translated_text)  # Print each translated chunk
    
    end_time = start_time + len(chunk) / RATE
    return start_time, end_time, translated_text

async def stream_processing():
    audio_queue = queue.Queue()
    translated_chunks = []

    capture_thread = threading.Thread(target=capture_audio, args=(audio_queue,))
    capture_thread.start()

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        start_time = time.time()
        while capture_thread.is_alive() or not audio_queue.empty():
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                tasks.append(loop.run_in_executor(executor, process_chunk, audio_chunk, start_time))
                start_time += len(audio_chunk) / RATE

        translated_chunks = await asyncio.gather(*tasks)

    # Create SRT file with specified path
    srt_file_path = "transcript.srt"  # Change this to your desired path
    create_srt([(format_time(start), format_time(end), text) for start, end, text in translated_chunks], filename=srt_file_path)

    final_translation = " ".join([text for _, _, text in translated_chunks])
    print("Combined Translation:", final_translation)  # Print the final combined translation
    return final_translation

# Example usage
translated_text = asyncio.run(stream_processing())
print("Combined Translation:", translated_text)  # Print the final combined translation