from functools import lru_cache
import gzip
import bz2
import lzma
import zstandard
from pathlib import Path
from log_utils import log

cs_cache = {}
# @lru_cache(maxsize=32)
def get_compressed_size(data_bytes: bytes, compressor_name: str, id: str | None) -> int:
    if id and id in cs_cache:
        return cs_cache[(id, compressor_name)]
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
        cs_cache[(id, compressor_name)] = len(compressed_data)
    return len(compressed_data)

def calculate_ncd_from_file_paths(signature_file_x_path: Path, signature_file_y_path: Path, compressor_name: str) -> float | None:
    try:
        with open(signature_file_x_path, 'rb') as f:
            data_x = f.read()
        with open(signature_file_y_path, 'rb') as f:
            data_y = f.read()

        return calculate_ncd_from_data(data_x, str(signature_file_x_path), data_y, str(signature_file_y_path), compressor_name)

    except FileNotFoundError:
        log("ERROR", f"Error: One or both signature files not found: {signature_file_x_path}, {signature_file_y_path}")
        return None
    except Exception as e:
        log("ERROR", f"Error calculating NCD for {signature_file_x_path} and {signature_file_y_path} using {compressor_name}: {e}")
        return None

def calculate_ncd_from_data(data_x: bytes, x_file_path: str, data_y: bytes, y_file_path: str, compressor_name: str) -> float | None:
    try:
        c_x = get_compressed_size(data_x, compressor_name, x_file_path)
        c_y = get_compressed_size(data_y, compressor_name, y_file_path)

        # Concatenate data_x and data_y for C(xy)
        data_xy = data_x + data_y
        c_xy = get_compressed_size(data_xy, compressor_name, None)

        min_c = min(c_x, c_y)
        max_c = max(c_x, c_y)

        if max_c == 0:
            return 0.0 if c_xy == 0 else 1.0

        ncd = (c_xy - min_c) / max_c
        return ncd

    except Exception as e:
        log("ERROR", f"Error calculating NCD from data: {x_file_path} vs {y_file_path} using {compressor_name}: {e}")
        return None