from pathlib import Path
import os
import shutil
from log_utils import log

GETMAXFREQS_EXE = Path("./bin/GetMaxFreqs")
SOX_EXE = Path("sox")
if os.name == 'nt':
    GETMAXFREQS_EXE = Path("./bin/GetMaxFreqs.exe")  
    SOX_EXE = Path('C:\\Program Files (x86)\\sox-14-4-2\\sox.exe')
    if not shutil.which(str(SOX_EXE)):
        log("WARNING",f"sox.exe not found in {SOX_EXE}.")
        for dir in Path("C:\\Program Files (x86)").iterdir():
            if dir.is_dir() and dir.name.startswith("sox"):
                SOX_EXE = dir / "sox.exe"
                if shutil.which(str(SOX_EXE)):
                    log("WARNING",f"Using sox.exe from {SOX_EXE}.")
                    break

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
