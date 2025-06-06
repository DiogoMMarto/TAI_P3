# main_script.py
from math import exp
from pathlib import Path
import shutil
from typing import Tuple

import concurrent.futures

import config
import audio_utils
import compression_utils
from log_utils import log

def softmax(x):
    max_x = max(x)
    exp_x = [exp(i-max_x) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]

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
            success = audio_utils.process_audio_file_parallel(audio_file)                
            if not success:
                log("ERROR",f"Failed to generate signatures for {audio_file.name}")
        else:
            log("WARNING",f"Skipping non-audio file: {audio_file.name}")
    log("INFO","--- Database Signature Preparation Complete ---")

def calculate_ncd_worker(
    signature_file: Path,
    db_signature_file: Path,
    compressor: str
) -> Tuple[str, str, str, float ] | None:
    """
    Worker function to calculate NCD for a single pair of files.
    """
    try:
        ncd = compression_utils.calculate_ncd(
            signature_file, db_signature_file, compressor
        )
        db_signature_dir_name = db_signature_file.parent.name
        query_stem = signature_file.stem
        db_path = f"{db_signature_dir_name}/{db_signature_file.stem}"
        if ncd is None:
            log("ERROR", f"Failed to calculate NCD for {signature_file.name} vs {db_signature_file.name} with {compressor}.")
            return None
        return (compressor, query_stem, db_path, ncd)
    except Exception as e:
        log("ERROR", f"Failed NCD: {signature_file.name} vs "
                   f"{db_signature_file.name} with {compressor}. Error: {e}")
        return None

def identify_music(query_audio_path: Path,
                   use_segment: bool = True,
                   add_noise_flag: bool = False,
                   noise_params: dict | None = None
                   ):
    """
    Identifies a query audio by comparing its NCD against the database signatures
    """
    log("INFO",f"--- Identifying Music for Query: {query_audio_path.name} ---")
    if not query_audio_path.exists():
        log("ERROR",f"Query audio file not found: {query_audio_path}")
        return {}

    actual_query_file = query_audio_path

    if add_noise_flag and noise_params:
        noisy_segment_path = config.TEMP_DIR / f"{query_audio_path.stem}_noisy.wav"
        log("INFO",f"Adding noise to {query_audio_path.name}")
        if not audio_utils.add_noise(query_audio_path, noisy_segment_path, noise_params["noise_level"], noise_params["noise_type"]):
            log("ERROR","Failed to add noise.")
            return {}
        actual_query_file = noisy_segment_path

    segment_duration = config.SEGMENT_DURATION
    if not use_segment:
        segment_duration = 1e9

    signatures_of_query_dir = config.TEMP_DIR / f"signatures"
    if not audio_utils.process_audio_file_parallel(actual_query_file, 
                                          segment_duration=segment_duration, 
                                          database_signature_path=signatures_of_query_dir):
        log("ERROR","Failed to process audio file.")
        return {}

    results_by_compressor: dict[str,dict[str, list[tuple[str, float]]]] = {}

    tasks_to_submit = []

    query_files = list(signatures_of_query_dir.rglob("*.freqs"))
    db_files = []
    for db_signature_dir in config.DATABASE_SIGNATURES_DIR.iterdir():
        if db_signature_dir.is_dir():
            db_files.extend(list(db_signature_dir.rglob("*.freqs")))

    log("INFO", f"Found {len(query_files)} query files and {len(db_files)} DB files.")

    for compressor in config.COMPRESSORS:
        for signature_file in query_files:
            for db_signature_file in db_files:
                tasks_to_submit.append(
                    (signature_file, db_signature_file, compressor)
                )

    log("INFO", f"Total NCD calculations to perform: {len(tasks_to_submit)}")

    # Execute tasks in parallel
    # You can adjust max_workers. None usually means using all available CPU cores.
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        # Submit tasks and store futures
        future_to_task = {
            executor.submit(calculate_ncd_worker, *task): task
            for task in tasks_to_submit
        }
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            if result:
                compressor, query_stem, db_path, ncd = result
                results_by_compressor.setdefault(compressor, {}) \
                    .setdefault(query_stem, []) \
                    .append((db_path, ncd))

    for query_file in query_files:
        if query_file.exists():
            query_file.unlink()
    
    return results_by_compressor

def rank_results(results_by_compressor: dict[str,dict[str, list[tuple[str, float]]]]):
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
    
    # softmax the scores
    ranks = {}
    for compressor, results in ranked_results.items():
        scores = [(song,result[0]) for song,result in results.items()]
        softmax_scores = softmax([score[1] for score in scores])
        for song, score in zip(scores, softmax_scores):
            ranks.setdefault(compressor, {})[song[0]] = score
            
    # Sort the results
    for compressor, results in ranks.items():
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        ranks[compressor] = sorted_results
        
    return ranks
        
def cleanup_temp_files():
    """
    Cleans up temporary files created during processing.
    """
    log("DEBUG","Cleaning up temporary files...")
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