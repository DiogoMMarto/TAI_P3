# main_script.py
from pathlib import Path
from pprint import pprint
import shutil

import config
import audio_utils
import compression_utils
from log_utils import log

def prepare_database_signatures():
    """
    Generates signatures for all music files in the database directory
    Skips if signature already exists, unless force_regenerate is True.
    """
    log("INFO","--- Preparing Database Signatures ---")
    if not config.DATABASE_MUSIC_DIR.exists():
        log("ERROR",f"Database music directory not found: {config.DATABASE_MUSIC_DIR}")
        return

    for audio_file in config.DATABASE_MUSIC_DIR.iterdir():
        if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3']:
            success = audio_utils.process_audio_file(audio_file)                
            if not success:
                log("ERROR",f"Failed to generate signatures for {audio_file.name}")
        else:
            log("WARNING",f"Skipping non-audio file: {audio_file.name}")
    log("INFO","--- Database Signature Preparation Complete ---")

def identify_music(query_audio_path: Path,
                   use_segment: bool = True,
                   add_noise_flag: bool = False,
                   noise_params: dict | None = None
                   ):
    """
    Identifies a query audio by comparing its NCD against the database signatures
    """
    log("INFO",f"\n--- Identifying Music for Query: {query_audio_path.name} ---")
    if not query_audio_path.exists():
        log("ERROR",f"Query audio file not found: {query_audio_path}")
        return {}

    actual_query_file = query_audio_path

    if add_noise_flag:
        noisy_segment_path = config.TEMP_DIR / f"{query_audio_path.stem}_noisy.wav"
        log("INFO",f"Adding noise to {query_audio_path.name}")
        if not audio_utils.add_noise(query_audio_path, noisy_segment_path, noise_params):
            log("ERROR","Failed to add noise.")
            return {}
        actual_query_file = noisy_segment_path

    segment_duration = config.SEGMENT_DURATION
    if not use_segment:
        segment_duration = 1e9

    signatures_of_query_dir = config.TEMP_DIR / f"signatures"
    if not audio_utils.process_audio_file(actual_query_file, 
                                          segment_duration=segment_duration, 
                                          database_signature_path=signatures_of_query_dir):
        log("ERROR","Failed to process audio file.")
        return {}

    results_by_compressor: dict[dict[str, list[str, float]]] = {}

    for compressor in config.COMPRESSORS: 
        log("INFO",f"Using Compressor: {compressor}")
        for signature_file in signatures_of_query_dir.rglob("*.freqs"):
            for db_signature_dir in config.DATABASE_SIGNATURES_DIR.iterdir():
                for db_signature_file in db_signature_dir.rglob("*.freqs"):
                    ncd = compression_utils.calculate_ncd(signature_file, db_signature_file, compressor)
                    results_by_compressor.setdefault(compressor, {}) \
                        .setdefault(signature_file.stem, []) \
                        .append((db_signature_dir.name + "/" + db_signature_file.stem, ncd))

    return results_by_compressor

def rank_results(results_by_compressor: dict[dict[str, list[str, float]]]):
    """
    Ranks the results based on NCD values.
    """
    log("INFO","--- Ranking Results ---")
    ranked_results = {}
    for compressor, results in results_by_compressor.items():
        ranked_results[compressor] = {}
        for _ , list_of_results in results.items():
            list_of_results.sort(key=lambda x: x[1])
            name , ncd = list_of_results[0]
            song_name , _ = name.split("/")[-2], name.split("/")[-1]
            cur_score = ranked_results.get(compressor, {}).get(song_name, (0,0))
            number_of_best_segments = cur_score[0] + 1
            avg_ncd = (cur_score[1] * cur_score[0] + ncd) / number_of_best_segments
            ranked_results[compressor][song_name] = (number_of_best_segments, avg_ncd)
    return ranked_results
        
def cleanup_temp_files():
    """
    Cleans up temporary files created during processing.
    """
    log("INFO","Cleaning up temporary files...")
    shutil.rmtree(config.TEMP_DIR)

def main():
    """
    Main function to run the script.
    # """
    log("INFO","--- Starting Music Identification Script ---")
    prepare_database_signatures()
    for query_file_path in config.QUERY_SAMPLES_DIR.iterdir():
        if query_file_path.suffix.lower() in ['.wav', '.flac', '.mp3']:
            log("INFO",f"Processing query file: {query_file_path.name}")
            ranks = identify_music(query_file_path)
            p = rank_results(ranks)
            log("INFO",f"Ranked results for {query_file_path.name}: {p}")
        else:
            log("WARNING",f"Skipping non-audio file: {query_file_path.name}")
    cleanup_temp_files()
    log("INFO","--- Script Finished ---")
    
if __name__ == "__main__":
    main()