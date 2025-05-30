from math import exp
from pathlib import Path
import shutil
import concurrent.futures
import config
import audio_utils
import compression_utils
from log_utils import log

def softmax(x):
    """Apply softmax normalization"""
    if not x:
        return []
    max_x = max(x)
    exp_x = [exp(i - max_x) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x] if sum_exp_x > 0 else [1.0/len(x) for _ in x]

def process_database_file(audio_file):
    """Worker function for database processing"""
    try:
        success = audio_utils.process_audio_file_parallel(audio_file, max_workers=16)
        return (audio_file.name, success)
    except Exception as e:
        log("ERROR", f"Exception processing {audio_file.name}: {e}")
        return (audio_file.name, False)
        
def prepare_database_signatures_parallel(max_workers: int = 6):
    """Parallel database signature preparation"""
    log("INFO", "--- Preparing Database Signatures (Parallel) ---")
    
    if not config.DATABASE_MUSIC_DIR.exists():
        log("ERROR", f"Database directory not found: {config.DATABASE_MUSIC_DIR}")
        return
    
    audio_files = [f for f in config.DATABASE_MUSIC_DIR.iterdir() 
                  if f.suffix.lower() in ['.wav', '.flac', '.mp3']]
    
    if not audio_files:
        log("WARNING", "No audio files found in database directory")
        return
    
    log("INFO", f"Processing {len(audio_files)} audio files in parallel")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_database_file, audio_files))
    
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    log("INFO", f"Database preparation complete: {successful} successful, {failed} failed")

def calculate_ncd_multi_compressor_worker(args):
    """Worker for parallel NCD calculation across multiple compressors"""
    signature_file, db_signature_file, compressors = args
    results = {}
    
    for compressor in compressors:
        try:
            ncd = compression_utils.calculate_ncd(signature_file, db_signature_file, compressor)
            results[compressor] = ncd
        except Exception as e:
            log("ERROR", f"NCD calculation failed for {compressor}: {e}")
            results[compressor] = None
    
    return signature_file, db_signature_file, results

def identify_music_parallel(query_audio_path: Path,
                          use_segment: bool = True,
                          add_noise_flag: bool = False,
                          noise_params: dict | None = None,
                          max_workers: int = 8) -> dict:
    """Parallel music identification"""
    log("INFO", f"--- Identifying Music (Parallel): {query_audio_path.name} ---")
    
    if not query_audio_path.exists():
        log("ERROR", f"Query file not found: {query_audio_path}")
        return {}

    actual_query_file = query_audio_path

    # Add noise if requested
    if add_noise_flag:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        noisy_path = config.TEMP_DIR / f"{query_audio_path.stem}_noisy.wav"
        log("INFO", f"Adding noise to {query_audio_path.name}")
        
        if noise_params is None:
            noise_params = {"noise_type": "whitenoise", "noise_level": 0.005}
            
        if not audio_utils.add_noise(query_audio_path, noisy_path, **noise_params):
            log("ERROR", "Failed to add noise")
            return {}
        actual_query_file = noisy_path

    # Process query audio
    segment_duration = config.SEGMENT_DURATION if use_segment else 1e9
    signatures_dir = config.TEMP_DIR / "signatures" / query_audio_path.stem
    
    if not audio_utils.process_audio_file_parallel(actual_query_file, 
                                                  segment_duration=segment_duration,
                                                  database_signature_path=signatures_dir,
                                                  max_workers=16):
        log("ERROR", "Failed to process query audio")
        return {}

    # Get signature files
    query_files = list(signatures_dir.rglob("*.freqs"))
    db_files = []
    
    for db_dir in config.DATABASE_SIGNATURES_DIR.iterdir():
        if db_dir.is_dir():
            db_files.extend(list(db_dir.rglob("*.freqs")))

    if not db_files:
        log("ERROR", "No database signature files found")
        return {}

    log("INFO", f"Found {len(query_files)} query files and {len(db_files)} DB files")

    # Prepare tasks for parallel NCD computation
    ncd_tasks = [(qf, dbf, config.COMPRESSORS) for qf in query_files for dbf in db_files]
    
    log("INFO", f"Total NCD task groups: {len(ncd_tasks)}")
    
    # Execute NCD calculations in parallel
    results_by_compressor = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for signature_file, db_signature_file, compressor_results in executor.map(
            calculate_ncd_multi_compressor_worker, ncd_tasks):
            
            query_stem = signature_file.stem
            db_path = f"{db_signature_file.parent.name}/{db_signature_file.stem}"
            
            for compressor, ncd in compressor_results.items():
                if ncd is not None:
                    results_by_compressor.setdefault(compressor, {}) \
                        .setdefault(query_stem, []) \
                        .append((db_path, ncd))

    # Cleanup
    try:
        if signatures_dir.exists():
            shutil.rmtree(signatures_dir)
        if add_noise_flag and actual_query_file != query_audio_path:
            if actual_query_file.exists():
                actual_query_file.unlink()
    except Exception as e:
        log("WARNING", f"Cleanup failed: {e}")
    
    log("INFO", f"Parallel identification complete: {len(results_by_compressor)} compressors")
    return results_by_compressor

def rank_results_parallel(results_by_compressor: dict) -> dict:
    """Parallel result ranking"""
    log("INFO", "--- Ranking Results (Parallel) ---")
    
    def rank_compressor_results(compressor_data):
        """Worker function for ranking results of a single compressor"""
        compressor, results = compressor_data
        ranked_results = {}
        
        for query_segment, list_of_results in results.items():
            if not list_of_results:
                continue
                
            # Sort by NCD (lower is better)
            list_of_results.sort(key=lambda x: x[1])
            best_match, best_ncd = list_of_results[0]
            
            # Extract song name
            song_name = best_match.split("/")[0] if "/" in best_match else best_match
            
            # Update statistics
            current_stats = ranked_results.get(song_name, (0, 0.0))
            num_segments = current_stats[0] + 1
            avg_ncd = (current_stats[1] * current_stats[0] + best_ncd) / num_segments
            ranked_results[song_name] = (num_segments, avg_ncd)
        
        return compressor, ranked_results
    
    # Parallel ranking across compressors
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        ranked_by_compressor = dict(executor.map(rank_compressor_results, 
                                                results_by_compressor.items()))
    
    # Apply softmax normalization
    final_ranks = {}
    for compressor, results in ranked_by_compressor.items():
        if not results:
            final_ranks[compressor] = []
            continue
            
        songs_and_counts = [(song, stats[0]) for song, stats in results.items()]
        segment_counts = [count for _, count in songs_and_counts]
        softmax_scores = softmax(segment_counts)
        
        song_scores = [(song, score) for (song, _), score in zip(songs_and_counts, softmax_scores)]
        song_scores.sort(key=lambda x: x[1], reverse=True)
        final_ranks[compressor] = song_scores
        
    return final_ranks

def process_single_query(query_path):
        """Worker function for single query processing"""
        try:
            log("INFO", f"Processing query: {query_path.name}")
            results = identify_music_parallel(query_path)
            
            if results:
                ranked_results = rank_results_parallel(results)
                return query_path.name, ranked_results
            else:
                return query_path.name, {}
                
        except Exception as e:
            log("ERROR", f"Error processing {query_path.name}: {e}")
            return query_path.name, {}

def process_multiple_queries_parallel(query_paths: list, max_workers: int = 16):
    """Process multiple query files in parallel"""
    log("INFO", f"Processing {len(query_paths)} queries in parallel")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_query, query_paths))
    
    return dict(results)

def main():
    """Enhanced main function with full parallelization"""
    log("INFO", "--- Starting Parallel Music Identification System ---")
    
    try:
        # Parallel database preparation
        prepare_database_signatures_parallel(max_workers=16)
        
        # Get query files
        if not config.QUERY_SAMPLES_DIR.exists():
            log("ERROR", f"Query directory not found: {config.QUERY_SAMPLES_DIR}")
            return
        
        query_files = [f for f in config.QUERY_SAMPLES_DIR.iterdir() 
                      if f.suffix.lower() in ['.wav', '.flac', '.mp3']]
        
        if not query_files:
            log("WARNING", "No query files found")
            return
        
        # Process all queries in parallel
        all_results = process_multiple_queries_parallel(query_files, max_workers=16)
        
        # Display results
        for query_name, ranked_results in all_results.items():
            log("INFO", f"Results for {query_name}:")
            for compressor, rankings in ranked_results.items():
                if rankings:
                    top_match = rankings[0]
                    log("INFO", f"  {compressor}: {top_match[0]} (score: {top_match[1]:.4f})")
                else:
                    log("INFO", f"  {compressor}: No results")
                    
    except Exception as e:
        log("ERROR", f"Fatal error: {e}")
    finally:
        # Cleanup
        try:
            if config.TEMP_DIR.exists():
                shutil.rmtree(config.TEMP_DIR)
        except Exception as e:
            log("WARNING", f"Cleanup failed: {e}")
        log("INFO", "--- Parallel Processing Complete ---")

if __name__ == "__main__":
    main()