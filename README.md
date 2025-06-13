# Music Identification using Compression-Based Signatures

This project implements a music identification system using **Normalized Compression Distance (NCD)** computed on frequency-domain representations of audio files. It includes methods for preprocessing audio, extracting features, generating signatures, performing similarity searches, and evaluating robustness under different noise levels.

---
## Motivation

The goal is to assess how well **compression-based distance metrics** can identify songs by comparing them to a known database of tracks. The idea builds upon **Kolmogorov complexity** by approximating similarity using real-world compressors applied to structured representations of audio.

---

## Methodology

### 1. Preprocessing and Feature Extraction
- **Input**: FLAC audio files.
- **Process**:
  - Convert audio to mono and resample.
  - Apply **Short-Time Fourier Transform (STFT)** to obtain a spectrogram.
  - Compute the **magnitude spectrum**.
  - Normalize and quantize to reduce variability and size.
- **Output**: Compact frequency representation per file.

### 2. Signature Generation
- Compress the processed frequency data using various **lossless compressors** (e.g., gzip, bzip2, lzma).
- Store the compressed outputs as "signatures" for each song in a database.

### 3. Similarity Search using NCD
- For a query song:
  - Apply the same preprocessing and signature extraction.
  - Compute NCD between the query signature and each signature in the database.
  - Use:  
    ```
    NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    ```
- To accelerate, use **Annoy** (Approximate Nearest Neighbors) to narrow down the comparison set based on frequency vectors.

### 4. Ranking Strategies
Use different ways to rank matches:
- **Basic**: Sort by lowest NCD.
- **Statistical**: Use mean, median, min, max of NCDs across compressors.
- **Weighted**: Apply softmax or exponential weighting.
- **Decay & Variance (DV)**: Penalize based on NCD spread and ranking volatility.

---

## Experimental Setup

### Noise Robustness Evaluation
- Noise types: `white`, `pink`, and `brown`.
- Noise levels: from `0.1` to `1.0`.
- Query songs are synthetically noised versions of known tracks.
- For each level:
  - Identify song using the full pipeline.
  - Compute accuracy and F1 score per compressor.

### Metrics
- **Accuracy** and **Weighted F1-score** computed using `sklearn`.
- Evaluated across compressors and noise levels.

---

## Results

The results of the noise robustness evaluation can be found in the `results/` directory.  
This includes:

- Accuracy and F1-score plots across different noise levels.
- A text summary file (`results.txt`) listing all performance metrics.

---

## How to Run

### 1. Install Dependencies

Install the required packages by running:

```bash
pip install -r requirements.txt
```

### 2. Run the System

To process and evaluate music identification with noisy queries:

```bash
python3 src/main.py
```

To run the full test script for robustness against noise:

```bash
python3 test_identification.py --noise xx xx xx --noise_type xxx --results_dir xxx
```
---

# Songs

- Used songs from www.classicals.de
