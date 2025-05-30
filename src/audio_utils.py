import subprocess
import shutil
from pathlib import Path
import config
from log_utils import log

def generate_signature(audio_path: Path, signature_path: Path, verbose: bool = False,
                       ws: int = config.GMF_WINDOW_SIZE,
                       sh: int = config.GMF_SHIFT,
                       ds: int = config.GMF_DOWNSAMPLING,
                       nf: int = config.GMF_NUM_FREQS) -> bool:
    """
    Generates a frequency signature file from an audio file using GetMaxFreqs.
    Returns True on success, False on failure.
    """
    if not config.GETMAXFREQS_EXE.exists():
        log("ERROR",f"Error: GetMaxFreqs executable not found at {config.GETMAXFREQS_EXE}")
        return False
    if not audio_path.exists():
        log("ERROR",f"Error: Audio file not found at {audio_path}")
        return False

    cmd = [
        str(config.GETMAXFREQS_EXE),
        "-w", str(signature_path),
        "-ws", str(ws),
        "-sh", str(sh),
        "-ds", str(ds),
        "-nf", str(nf),
        str(audio_path)
    ]
    if verbose:
        cmd.insert(1, "-v")
    log("DEBUG",f"Running GetMaxFreqs: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose and process.stdout:
            log("INFO","GetMaxFreqs STDOUT:", process.stdout)
        if process.stderr:
            log("ERROR","GetMaxFreqs STDERR:", process.stderr)
            return False
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR",f"Error running GetMaxFreqs for {audio_path}:")
        log("ERROR","Command:", ' '.join(e.cmd))
        log("ERROR","Return code:", e.returncode)
        log("ERROR","STDOUT:", e.stdout)
        log("ERROR","STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        log("ERROR",f"Error: GetMaxFreqs command '{config.GETMAXFREQS_EXE}' not found. Is it in your PATH or correctly specified in config.py?")
        return False

def extract_segment(input_audio_path: Path, output_segment_path: Path,
                    start_time: float, duration: float) -> bool:
    """
    Extracts a segment from an audio file using SoX.
    Returns True on success, False on failure.
    """
    if not shutil.which(str(config.SOX_EXE)):
        log("ERROR",f"Error: SoX executable not found at '{config.SOX_EXE}'. Please install SoX or check config.py.")
        return False
    if not input_audio_path.exists():
        log("ERROR",f"Error: Input audio file not found at {input_audio_path}")
        return False

    cmd = [
        str(config.SOX_EXE),
        str(input_audio_path),
        str(output_segment_path),
        "trim", str(start_time), str(duration)
    ]
    log("DEBUG",f"Running SoX (extract_segment): {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR",f"Error running SoX for segment extraction on {input_audio_path}:")
        log("ERROR","STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        log("ERROR",f"Error: SoX command '{config.SOX_EXE}' not found.")
        return False

def convert_audio_format(input_audio_path: Path, output_audio_path: Path) -> bool:
    """
    Converts an audio file to WAV format using SoX. with sample rate 44100 Hz.
    This is a workaround for the GetMaxFreqs requirement of WAV format.
    Returns True on success, False on failure.
    """
    if not shutil.which(str(config.SOX_EXE)):
        log("ERROR",f"Error: SoX executable not found at '{config.SOX_EXE}'. Please install SoX or check config.py.")
        return False
    if not input_audio_path.exists():
        log("ERROR",f"Error: Input audio file not found at {input_audio_path}")
        return False
    if not output_audio_path.parent.exists():
        log("ERROR",f"Error: Output directory for {output_audio_path} does not exist.")
        return False

    cmd = [
        str(config.SOX_EXE),
        str(input_audio_path),
        str(output_audio_path),
        "rate", "44100",
        "channels", "2",
    ]

    log("DEBUG",f"Running SoX (convert_audio_format): {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR",f"Error running SoX for audio format conversion on {input_audio_path}:")
        log("ERROR","STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        log("ERROR",f"Error: SoX command '{config.SOX_EXE}' not found.")
        return False

def add_noise(input_audio_path: Path, output_noisy_audio_path: Path,
              noise_type: str = "whitenoise", noise_level: float = 0.005) -> bool:
    """
    Adds noise to an audio file using SoX, ensuring sample rate and channel compatibility.
    Example noise_types: 'whitenoise', 'pinknoise', 'brownnoise'
    Returns True on success, False on failure.
    """
    if not shutil.which(str(config.SOX_EXE)):
        log("ERROR", f"Error: SoX executable not found at '{config.SOX_EXE}'. Please install SoX or check config.py.")
        return False
    if not input_audio_path.exists():
        log("ERROR", f"Error: Input audio file not found at {input_audio_path}")
        return False
    if not output_noisy_audio_path.parent.exists():
        log("ERROR", f"Error: Output directory for {output_noisy_audio_path} does not exist.")
        return False

    duration = 0.0
    sample_rate = None
    channels = None
    try:
        info_duration_cmd = [str(config.SOX_EXE), "--i", "-D", str(input_audio_path)]
        duration_str = subprocess.run(info_duration_cmd, check=True, capture_output=True, text=True).stdout.strip()
        duration = float(duration_str)

        info_full_cmd = [str(config.SOX_EXE), "--info", str(input_audio_path)]
        info_output = subprocess.run(info_full_cmd, check=True, capture_output=True, text=True).stdout

        for line in info_output.splitlines():
            if "Sample Rate" in line:
                sample_rate = int(line.split(":")[-1].strip().split()[0])
            elif "Channels" in line:
                channels = int(line.split(":")[-1].strip())
        
        if sample_rate is None or channels is None:
            log("ERROR", f"Could not determine sample rate or channels for {input_audio_path}")
            return False

    except Exception as e:
        log("ERROR", f"Could not get audio properties for {input_audio_path}: {e}")
        return False

    log("INFO", f"Input audio properties for {input_audio_path}: Duration={duration}s, Sample Rate={sample_rate}Hz, Channels={channels}")

    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_noise_file = config.TEMP_DIR / f"temp_noise_{input_audio_path.stem}.wav"

    cmd_gen_noise = [
        str(config.SOX_EXE), "-n", str(temp_noise_file),
        "synth", str(duration), noise_type,
        "vol", str(noise_level),
        "rate", str(sample_rate),
        "channels", str(channels)  
    ]
    cmd_mix_noise = [
        str(config.SOX_EXE), "-m", str(input_audio_path), str(temp_noise_file), str(output_noisy_audio_path)
    ]

    log("INFO", f"Running SoX (generate noise): {' '.join(cmd_gen_noise)}")
    try:
        subprocess.run(cmd_gen_noise, check=True, capture_output=True, text=True)
        log("INFO", f"Running SoX (mix noise): {' '.join(cmd_mix_noise)}")
        subprocess.run(cmd_mix_noise, check=True, capture_output=True, text=True)
        
        if temp_noise_file.exists():
            temp_noise_file.unlink() 
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Error running SoX for noise addition on {input_audio_path}:")
        log("ERROR", "STDOUT:", e.stdout)
        log("ERROR", "STDERR:", e.stderr)
        if temp_noise_file.exists():
            temp_noise_file.unlink()
        return False
    except FileNotFoundError:
        log("ERROR", f"Error: SoX command '{config.SOX_EXE}' not found.")
        return False
    
def song_duration(input_audio_path: Path) -> float | None:
    """
    Returns the duration of an audio file in seconds.
    Returns None if the duration cannot be determined.
    """
    if not shutil.which(str(config.SOX_EXE)):
        log("ERROR",f"Error: SoX executable not found at '{config.SOX_EXE}'. Please install SoX or check config.py.")
        return None
    if not input_audio_path.exists():
        log("ERROR",f"Error: Input audio file not found at {input_audio_path}")
        return None

    cmd = [
        str(config.SOX_EXE),
        "--i",
        "-D",
        str(input_audio_path)
    ]
    try:
        duration_str = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        log("ERROR",f"Error running SoX for duration extraction on {input_audio_path}:")
        log("ERROR","STDERR:", e.stderr)
        return None
    
    
def process_audio_file(
    input_audio_path: Path, 
    segment_duration: float = config.SEGMENT_DURATION,
    database_signature_path: Path = config.DATABASE_SIGNATURES_DIR,
    temp_dir: Path = config.TEMP_DIR,
    ) -> bool:
    """
    Processes an audio file by converting it to WAV format and extact the signature of the segments.
    Returns True on success, False on failure.
    """
    output_audio_path = temp_dir / (input_audio_path.stem + ".wav")
    signature_path = database_signature_path / f"{input_audio_path.stem}"
    
    if signature_path.exists():
        log("DEBUG",f"Signature already exists for {input_audio_path}. Skipping processing.")
        return True
    
    if not convert_audio_format(input_audio_path, output_audio_path):
        log("ERROR",f"Error converting {input_audio_path} to WAV format.")
        return False
    
    song_duration_value = song_duration(output_audio_path)
    if song_duration_value is None:
        log("ERROR",f"Error getting duration for {output_audio_path}.")
        return False
    
    number_of_segments = int(song_duration_value // segment_duration)
    if song_duration_value % segment_duration > 0:
        number_of_segments += 1
    
    signature_path.mkdir(parents=True, exist_ok=True)
    for i in range(number_of_segments):
        start_time = i * segment_duration
        segment_path = temp_dir / f"{output_audio_path.stem}_segment_{i}.wav"
        signature_path = database_signature_path / f"{output_audio_path.stem}/segment_{i}.freqs" 
        if not extract_segment(output_audio_path, segment_path, start_time, segment_duration):
            log("ERROR",f"Error extracting segment {i} from {output_audio_path}.")
            return False
        if not generate_signature(segment_path, signature_path):
            log("ERROR",f"Error generating signature for {segment_path}.")
            return False
        if segment_path.exists():
            segment_path.unlink()
    return True
    