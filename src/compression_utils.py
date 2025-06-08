from functools import lru_cache
import gzip
import bz2
import lzma
import zstandard
from pathlib import Path
from log_utils import log

cs_cache = {}
# @lru_cache(maxsize=32)
def get_compressed_size(data_bytes: bytes, compressor_name: str , id: str | None) -> int:
    """
    Compresses data using the specified compressor and returns the size of the compressed data in bytes.
    Supported compressors: 'gzip', 'bzip2', 'lzma', 'zstd'.
    """
    if id and id in cs_cache:
        return cs_cache[id]
    compressed_data = None
    if compressor_name == "gzip":
        compressed_data = gzip.compress(data_bytes)
    elif compressor_name == "bzip2":
        compressed_data = bz2.compress(data_bytes)
    elif compressor_name == "lzma":
        compressed_data = lzma.compress(data_bytes)
    elif compressor_name == "zstd":
        cctx = zstandard.ZstdCompressor()
        compressed_data = cctx.compress(data_bytes)
    else:
        raise ValueError(f"Unsupported compressor: {compressor_name}. Supported: gzip, bzip2, lzma, zstd.")
    if id:
        cs_cache[id] = len(compressed_data)
    return len(compressed_data)

def calculate_ncd(signature_file_x_path: Path, signature_file_y_path: Path, compressor_name: str) -> float | None:
    """
    Calculates the Normalized Compression Distance (NCD) between two signature files. [cite: 2]
    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    Returns the NCD value, or None if an error occurs.
    """
    try:
        with open(signature_file_x_path, 'rb') as f:
            data_x = f.read()
        with open(signature_file_y_path, 'rb') as f:
            data_y = f.read()

        c_x = get_compressed_size(data_x, compressor_name, str(signature_file_x_path))
        c_y = get_compressed_size(data_y, compressor_name , str(signature_file_y_path))

        # Concatenate data_x and data_y for C(xy) [cite: 2]
        data_xy = data_x + data_y
        c_xy = get_compressed_size(data_xy, compressor_name , None)

        min_c = min(c_x, c_y)
        max_c = max(c_x, c_y)

        if max_c == 0:
            return 0.0 if c_xy == 0 else 1.0

        ncd = (c_xy - min_c) / max_c
        return ncd

    except FileNotFoundError:
        log("ERROR",f"Error: One or both signature files not found: {signature_file_x_path}, {signature_file_y_path}")
        return None
    except Exception as e:
        log("ERROR",f"Error calculating NCD for {signature_file_x_path} and {signature_file_y_path} using {compressor_name}: {e}")
        return None