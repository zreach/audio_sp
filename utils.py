import numpy as np
import math

def rms_to_dbfs(rms):
    return 20.0 * math.log10(max(1e-16, rms)) + 3.0103


def max_dbfs(sample_data):
    # Peak dBFS based on the maximum energy sample. Will prevent overdrive if used for normalization.
    return rms_to_dbfs(max(abs(np.min(sample_data)), abs(np.max(sample_data))))


def mean_dbfs(sample_data):
    return rms_to_dbfs(math.sqrt(np.mean(np.square(sample_data, dtype=np.float64))))


def gain_db_to_ratio(gain_db):
    return math.pow(10.0, gain_db / 20.0)


def normalize_audio(sample_data, dbfs=3.0103):
    return np.maximum(np.minimum(sample_data * gain_db_to_ratio(dbfs - max_dbfs(sample_data)), 1.0), -1.0)