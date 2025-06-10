import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from log_utils import log
from main import identify_music, prepare_database_signatures, rank_results, build_annoy_index
from config import QUERY_SAMPLES_DIR, TEMP_DIR, COMPRESSORS, DATABASE_SIGNATURES_DIR, DATABASE_MUSIC_DIR
from audio_utils import add_noise
import concurrent.futures

# Global variable to store the Annoy index (initialized per process)
_db_annoy_index = None
_db_files = None

def parse_noise_levels(noise_args):
    return np.arange(noise_args[0], noise_args[1], noise_args[2])

def initialize_worker_annoy_index():
    """Initializes the Annoy index for each worker process."""
    global _db_annoy_index
    global _db_files
    if _db_annoy_index is None:
        db_files = []
        for db_signature_dir in DATABASE_SIGNATURES_DIR.iterdir():
            if db_signature_dir.is_dir():
                db_files.extend(list(db_signature_dir.rglob("*.freqs")))
        
        log("INFO", f"Building Annoy index for {len(db_files)} DB files in worker process.")
        _db_annoy_index = build_annoy_index(db_files)
        _db_files = db_files

def process_query(query_file: Path, noise_type, noise_level) -> dict[str, tuple[str, str]]:
    """
    Processes a single query file by adding noise and identifying the music.
    Assumes _db_annoy_index and _db_files are initialized in the worker process.
    """
    global _db_annoy_index
    global _db_files

    if _db_annoy_index is None or _db_files is None:
        # This should ideally not happen if initializer is set up correctly
        log("ERROR", "Annoy index not initialized in worker process.")
        return {}

    noisy_query = TEMP_DIR / f"noisy_{query_file.stem}_{noise_type}_{noise_level}.flac"
    log("INFO", f"Processing query file: {query_file.name} with noise type: {noise_type} and noise level: {noise_level}")
    
    # Add noise to the query file
    add_noise(query_file, 
              noisy_query, 
              noise_type=noise_type,
              noise_level=noise_level)
    
    # Identify the music using the already initialized Annoy index and db_files
    ranks = identify_music(noisy_query, db_annoy_index=_db_annoy_index, db_files=_db_files)
    p = rank_results(ranks)
    log("INFO", f"Ranked results for {noisy_query.name}: {p}")
    
    # Extract top result for each compressor
    ret = {}
    for compressor, results in p.items():
        if results:
            song_name, ncd = results[0]
            ret[compressor] = (song_name, query_file.stem)
    
    # Clean up the noisy query file
    noisy_query.unlink()
    return ret

def main():
    arg_parser = argparse.ArgumentParser(description="Process some integers.")
    arg_parser.add_argument(
        "--noise",
        type=float,
        nargs=3,
        default=[0.1, 1.0, 0.1],
        metavar=("start", "end", "step"),
        help="Noise levels to process, e.g. --noise 0.1 1.0 0.1",
    )
    arg_parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        choices=["white", "pink", "brown"],
        help="Type of noise to add to the audio files.",
    )
    arg_parser.add_argument(
        "--results_dir",
        type=str,
        default=str("results"),
        help="Directory to save the results.",
    )
    
    args = arg_parser.parse_args()
    noise_levels = parse_noise_levels(args.noise)
    log("INFO", f"Processing noise levels: {noise_levels}")
    log("INFO","--- Starting Music Identification Script ---")
    
    # Prepare database signatures (this can still be done once in the main process)
    prepare_database_signatures()
    
    accuracies_by_compressor: dict[str,list] = {compressor: [] for compressor in COMPRESSORS}
    f1_scores_by_compressor: dict[str,list] = {compressor: [] for compressor in COMPRESSORS}
    
    for noise_level in noise_levels:
        log("INFO", f"Processing noise level: {noise_level}")
        y_true_by_compressors = {compressor: [] for compressor in COMPRESSORS}
        y_pred_by_compressors = {compressor: [] for compressor in COMPRESSORS}

        # Use initializer to set up Annoy index in each worker process
        with concurrent.futures.ProcessPoolExecutor(initializer=initialize_worker_annoy_index) as executor:
            dir_ = DATABASE_MUSIC_DIR
            dir_list = list(dir_.iterdir())
        
            future_to_tasks = [
                executor.submit(process_query, query_file, args.noise_type, noise_level)
                for query_file in dir_list
            ]
            
            for future in concurrent.futures.as_completed(future_to_tasks):
                try:
                    result = future.result()
                    if result:
                        for compressor, (song_name, query_file_stem) in result.items():
                            y_true_by_compressors[compressor].append(song_name)
                            y_pred_by_compressors[compressor].append(query_file_stem)
                except Exception as e:
                    log("ERROR", f"Error processing query: {e}")
            
        # Compute accuracy and F1 score for each compressor
        for compressor in COMPRESSORS:
            y_true = y_true_by_compressors[compressor]
            y_pred = y_pred_by_compressors[compressor]

            if not y_true or not y_pred:
                log("WARNING", f"No results for compressor {compressor} at noise level {noise_level}.")
                continue

            report: dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0) # type: ignore
            accuracies_by_compressor[compressor].append(report["accuracy"])
            f1_scores_by_compressor[compressor].append(report["weighted avg"]["f1-score"])

    os.makedirs(args.results_dir, exist_ok=True)
    
    # Plot accuracy and F1 score
    for compressor in COMPRESSORS:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(noise_levels, accuracies_by_compressor[compressor], marker='o')
        plt.title(f'Accuracy for {compressor}')
        plt.xlabel('Noise Level')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(noise_levels, f1_scores_by_compressor[compressor], marker='o', color='orange')
        plt.title(f'F1 Score for {compressor}')
        plt.xlabel('Noise Level')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.05)
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/{compressor}_noise_analysis.png")
        log("INFO", f"Saved plots for {compressor} in {args.results_dir}")
        plt.close()
    
    with open(f"{args.results_dir}/results.txt", "w") as f:
        f.write("Noise Level Analysis Results:\n")
        for compressor in COMPRESSORS:
            f.write(f"\nCompressor: {compressor}\n")
            f.write(f"Accuracies: {accuracies_by_compressor[compressor]}\n")
            f.write(f"F1 Scores: {f1_scores_by_compressor[compressor]}\n")

if __name__ == "__main__":
    main()