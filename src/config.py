from pathlib import Path
import os

GETMAXFREQS_EXE = Path("./bin/GetMaxFreqs")
SOX_EXE = Path("sox")
if os.name == 'nt':
    GETMAXFREQS_EXE = Path("./bin/GetMaxFreqs.exe")  
    SOX_EXE = Path("C:\\Users\\D\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\sox-14.4.2\\sox.exe")

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_MUSIC_DIR = BASE_DIR / "database"
DATABASE_SIGNATURES_DIR = BASE_DIR / "signatures"
QUERY_SAMPLES_DIR = BASE_DIR / "query"
TEMP_DIR = BASE_DIR / "temp" 

# --- GetMaxFreqs Parameters (from C++ defaults) ---
GMF_WINDOW_SIZE = 1024    # -ws option
GMF_SHIFT = 256           # -sh option
GMF_DOWNSAMPLING = 4      # -ds option
GMF_NUM_FREQS = 4         # -nf option

# --- Compressors ---
COMPRESSORS = ["gzip", "bzip2", "lzma", "zstd"]

# --- Parameters ---
SEGMENT_DURATION = 10

# --- Ensure directories exist ---
DATABASE_SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
QUERY_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
LOG_LEVEL = "INFO"