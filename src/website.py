from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from pathlib import Path
import shutil
import uvicorn
from tempfile import NamedTemporaryFile # Still useful for generic temp files, but we'll use Path.open() for specific naming
import os
from pydub import AudioSegment # Required for audio conversion
from pydub.utils import get_prober_name # For debugging ffmpeg/pydub setup
from uuid import uuid4 # For unique filenames

import config # Import your configuration module
from main import identify_music, build_annoy_index, rank_results_DV, rank_results, prepare_database_signatures # Import your music recognition functions

app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for database and Annoy index
db_files = []
db_annoy_index = None

# Load database and build Annoy index on startup
@app.on_event("startup")
async def startup_event():
    global db_files, db_annoy_index
    
    # Ensure the TEMP_DIR exists
    if not config.TEMP_DIR.exists():
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created TEMP_DIR: {config.TEMP_DIR}")

    # Optional: If you're having persistent issues with ffmpeg not being found
    # despite being in PATH, you can explicitly set its path here.
    # Replace "C:/path/to/your/ffmpeg/bin" with the actual path to your ffmpeg.exe
    # For example: set_paths_once(ffmpeg="C:/Users/YourUser/Downloads/ffmpeg-6.0-full_build/bin/ffmpeg.exe")
    # For systems where ffmpeg might be split (like Linux/macOS), you might also
    # need ffprobe: set_paths_once(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
    # set_paths_once(ffmpeg="C:/path/to/your/ffmpeg/bin/ffmpeg.exe") # Uncomment and modify if needed
    prepare_database_signatures()
    print(f"pydub using prober: {get_prober_name()}") # This will tell you if pydub found ffmpeg or avconv
    print("Loading music database and building Annoy index...")
    if not config.DATABASE_SIGNATURES_DIR.exists():
        print(f"Warning: config.DATABASE_SIGNATURES_DIR '{config.DATABASE_SIGNATURES_DIR}' does not exist. Skipping database loading.")
        return  
    for db_signature_dir in config.DATABASE_SIGNATURES_DIR.iterdir():
        if db_signature_dir.is_dir():
            db_files.extend(list(db_signature_dir.rglob("*.freqs")))
    print(f"Found {len(db_files)} signature files in the database.")
    db_annoy_index = build_annoy_index(db_files)
    print("Annoy index built.")
    


@app.post("/identify/") # Changed back to identify_song
async def identify_song(file: UploadFile = File(...)):
    """
    Endpoint to upload an audio file and identify the song.
    Handles various input audio formats by converting to WAV if necessary.
    """
    # Allowed input MIME types from browser recordings/uploads
    # Broadened list to include common browser recording types (webm, ogg with codecs)
    allowed_input_mimetypes = [
        "audio/wav", "audio/flac", "audio/mp3",
        "audio/webm", "audio/webm;codecs=opus",
        "audio/ogg", "audio/ogg;codecs=opus",
        "audio/mpeg" # Often used for MP3
    ]

    if file.content_type not in allowed_input_mimetypes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a supported audio file."
        )

    # Determine original file extension for temporary saving
    # Use the original filename suffix if available, otherwise guess from content_type
    original_suffix = Path(file.filename).suffix.lower() if Path(file.filename).suffix else ""
    if not original_suffix:
        if "webm" in file.content_type:
            original_suffix = ".webm"
        elif "ogg" in file.content_type:
            original_suffix = ".ogg"
        elif "mp3" in file.content_type or "mpeg" in file.content_type:
            original_suffix = ".mp3"
        elif "flac" in file.content_type:
            original_suffix = ".flac"
        elif "wav" in file.content_type:
            original_suffix = ".wav"
        else:
            original_suffix = ".bin" # Fallback

    # Generate a unique filename using UUID and the determined suffix
    unique_filename = f"{uuid4()}{original_suffix}"
    temp_input_file_path = config.TEMP_DIR / unique_filename
    temp_processed_file_path = None # Path for converted file if needed

    try:
        # Save the uploaded file to the specified temporary directory
        with open(temp_input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        await file.close() # CRITICAL: Explicitly close the UploadFile's stream

        print(f"Received file: {file.filename}, Content-Type: {file.content_type}, saved to {temp_input_file_path}")

        audio_to_process_path = temp_input_file_path
        # Check if conversion is needed for the recognition library
        # Conversion is needed if the backend recognition library ONLY accepts WAV/FLAC/MP3
        # and the incoming file is a browser-specific format like WebM or Ogg.
        if file.content_type not in ["audio/wav"]:
            print(f"Converting {file.content_type} to WAV for processing...")
            try:
                audio = AudioSegment.from_file(temp_input_file_path)
                # Create a new temporary file for the WAV output
                temp_processed_file_name = f"{uuid4()}.wav"
                temp_processed_file_path = config.TEMP_DIR / temp_processed_file_name
                # set dual channel with 444100 Hz sample rate
                audio = audio.set_channels(2).set_frame_rate(444100)
                audio.export(temp_processed_file_path, format="wav")
                # copy the file to the temp directory
                shutil.copy(temp_processed_file_path, config.TEMP_DIR / "A" / temp_processed_file_name)
                if not temp_processed_file_path.exists():
                    raise HTTPException(status_code=500, detail="Failed to convert audio file to WAV format.")
                audio_to_process_path = config.TEMP_DIR / "A" / temp_processed_file_name
                print(f"Converted to WAV: {temp_processed_file_path}")
            except Exception as e:
                # Provide a more informative error for conversion issues
                raise HTTPException(status_code=500, detail=f"Error converting audio file ({file.content_type}) to WAV: {e}. Ensure ffmpeg is installed and in your system's PATH.")

        results = identify_music(audio_to_process_path, db_annoy_index, db_files)
        ranked_results = rank_results(results)

        # Return the ranked results
        return JSONResponse(content={"ranked_results": ranked_results})

    except Exception as e:
        print(f"Error processing the file on server: {e}")
        # General exception handling for server-side issues
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up temporary files
        if temp_input_file_path and temp_input_file_path.exists():
            temp_input_file_path.unlink()
            print(f"Cleaned up input temp file: {temp_input_file_path}")
        if temp_processed_file_path and temp_processed_file_path.exists():
            temp_processed_file_path.unlink()
            print(f"Cleaned up processed temp file: {temp_processed_file_path}")
        if audio_to_process_path and audio_to_process_path.exists():
            audio_to_process_path.unlink()
            print(f"Cleaned up audio to process file: {audio_to_process_path}")


@app.get("/")
async def read_root():
    """
    Serve the homepage from the static directory.
    """
    index_file_path = Path("static/index.html")
    if not index_file_path.exists():
        raise HTTPException(status_code=404, detail="Homepage (static/index.html) not found.")
    return FileResponse(index_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

