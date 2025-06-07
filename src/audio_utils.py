import subprocess
import shutil
from pathlib import Path
import concurrent.futures
import config
from log_utils import log

def generate_signature(audio_path: Path, signature_path: Path, verbose: bool = False,
                       ws: int = config.GMF_WINDOW_SIZE,
                       sh: int = config.GMF_SHIFT,
                       ds: int = config.GMF_DOWNSAMPLING,
                       nf: int = config.GMF_NUM_FREQS) -> bool:
    """Generate frequency signature using GetMaxFreqs"""
    if not config.GETMAXFREQS_EXE.exists():
        log("ERROR", f"GetMaxFreqs executable not found at {config.GETMAXFREQS_EXE}")
        return False
    
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [str(config.GETMAXFREQS_EXE), "-w", str(signature_path),
           "-ws", str(ws), "-sh", str(sh), "-ds", str(ds), "-nf", str(nf),
           str(audio_path)]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"GetMaxFreqs failed for {audio_path}: {e.stderr}")
        return False


def extract_segment(input_audio_path: Path, output_segment_path: Path,
                    start_time: float, duration: float) -> bool:
    """Extract audio segment using SoX"""
    if not shutil.which(str(config.SOX_EXE)):
        log("ERROR", f"SoX executable not found at '{config.SOX_EXE}'")
        return False
    
    cmd = [str(config.SOX_EXE), str(input_audio_path), str(output_segment_path),
           "trim", str(start_time), str(duration)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"SoX segment extraction failed: {e.stderr}")
        return False

def convert_audio_format(input_audio_path: Path, output_audio_path: Path) -> bool:
    """Convert audio to WAV format using SoX"""
    cmd = [str(config.SOX_EXE), str(input_audio_path), str(output_audio_path),
           "rate", "44100", "channels", "2"]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Audio conversion failed: {e.stderr}")
        return False

def add_noise(input_audio_path: Path, output_noisy_audio_path: Path,
              noise_type: str = "whitenoise", noise_level: float = 0.005) -> bool:
    """Add noise to audio file using SoX"""
    # Implementation from the provided code
    duration = song_duration(input_audio_path)
    if duration is None:
        return False
    
    temp_noise_file = config.TEMP_DIR / f"temp_noise_{input_audio_path.stem}.wav"
    
    cmd_gen_noise = [str(config.SOX_EXE), "-n", str(temp_noise_file),
                     "synth", str(duration), noise_type, "vol", str(noise_level),
                     "rate", "44100", "channels", "2"]
    cmd_mix_noise = [str(config.SOX_EXE), "-m", str(input_audio_path), 
                     str(temp_noise_file), str(output_noisy_audio_path)]
    
    try:
        subprocess.run(cmd_gen_noise, check=True, capture_output=True, text=True)
        subprocess.run(cmd_mix_noise, check=True, capture_output=True, text=True)
        if temp_noise_file.exists():
            temp_noise_file.unlink()
        return True
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Noise addition failed: {e.stderr}")
        if temp_noise_file.exists():
            temp_noise_file.unlink()
        return False

def song_duration(input_audio_path: Path) -> float | None:
    """Get audio file duration using SoX"""
    cmd = [str(config.SOX_EXE), "--i", "-D", str(input_audio_path)]
    try:
        duration_str = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        log("ERROR", f"Duration extraction failed: {e.stderr}")
        return None
    
def process_audio_file_parallel(
    input_audio_path: Path, 
    segment_duration: float = config.SEGMENT_DURATION,
    database_signature_path: Path = config.DATABASE_SIGNATURES_DIR,
    temp_dir: Path = config.TEMP_DIR,
    max_workers: int = 8
) -> bool:
    """Process audio file with parallel segment processing"""
    log("INFO", f"Processing audio file in parallel: {input_audio_path.name}")
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_audio_path = temp_dir / (input_audio_path.stem + ".wav")
    signature_dir = database_signature_path / f"{input_audio_path.stem}"
    
    if signature_dir.exists() and any(signature_dir.glob("*.freqs")):
        log("DEBUG", f"Signatures already exist for {input_audio_path}")
        return True
    
    # Convert to WAV format
    if not convert_audio_format(input_audio_path, output_audio_path):
        log("ERROR", f"Error converting {input_audio_path} to WAV format")
        return False
    
    # Get duration and calculate segments
    song_duration_value = song_duration(output_audio_path)
    if song_duration_value is None:
        log("ERROR", f"Error getting duration for {output_audio_path}")
        return False
    
    number_of_segments = int(song_duration_value // segment_duration)
    if song_duration_value % segment_duration > 0:
        number_of_segments += 1
    
    log("INFO", f"Processing {number_of_segments} segments for {input_audio_path.name}")
    signature_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel processing
    segment_args = [
        (i, output_audio_path, temp_dir, database_signature_path, segment_duration) 
        for i in range(number_of_segments)
    ]
    
    # Process segments in parallel
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_segment_worker, segment_args))
        
        success_count = sum(results)
        log("INFO", f"Parallel processing: {success_count}/{len(results)} segments successful")
        
        # Clean up temporary WAV file
        if output_audio_path.exists():
            output_audio_path.unlink()
        
        return all(results)
        
    except Exception as e:
        log("ERROR", f"Exception during parallel processing: {e}")
        if output_audio_path.exists():
            output_audio_path.unlink()
        return False

def process_segment_worker(args):
    """Worker function to process a single audio segment in parallel"""
    i, output_audio_path, temp_dir, database_signature_path, segment_duration = args
    start_time = i * segment_duration
    segment_path = temp_dir / f"{output_audio_path.stem}_segment_{i}.wav"
    signature_path = database_signature_path / f"{output_audio_path.stem}/segment_{i}.freqs"
    
    try:
        if not extract_segment(output_audio_path, segment_path, start_time, segment_duration):
            log("ERROR", f"Failed to extract segment {i}")
            return False
        if not generate_signature(segment_path, signature_path):
            log("ERROR", f"Failed to generate signature for segment {i}")
            return False
        if segment_path.exists():
            segment_path.unlink()
        return True
    except Exception as e:
        log("ERROR", f"Exception processing segment {i}: {e}")
        if segment_path.exists():
            segment_path.unlink()
        return False