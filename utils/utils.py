import ast
import csv
from io import StringIO
import math
import os
import string
import warnings
import pathlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_FORMAT = "# {:.<60} {}"

POSSIBLE_TASKS = ["smatch", "batch", "experiment"]

MLFLOW_NOT_FOUND = False


def return_default_values(config):
    default_values = {
        "smoothing_method": "no",
        "backtrack": True,
        "training_algorithm": "word2vec",
        "write_walks": True,
        "flatten": "all",
        "epsilon": 0.1,
        "learning_method": "skipgram",
        "sentence_length": 60,
        "window_size": 5,
        "n_dimensions": 300,
        "sampling_factor": 0.001,
    }

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    return config


def _convert_to_bool(config, key):
    if config[key] in [True, False]:
        return config
    if config[key].lower() not in ["true", "false"]:
        raise ValueError(
            "Unknown {key} parameter {value}".format(key=key, value=config[key])
        )
    else:
        if config[key].lower() == "false":
            config[key] = False
        elif config[key].lower() == "true":
            config[key] = True
    return config

def _check_stream_configuration(config):
    """Validate configuration parameters"""
    required_params = [
        'kafka_topicid', 'bootstrap_servers', 'port',
        'window_strategy', 'update_frequency',
        'most_similar_inlist_n', 'output_format', 'kafka_groupid'
    ]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required configuration parameter: {param}")
        else:
            config["update_frequency"] = int(config["update_frequency"])
            config["most_similar_k"] = int(config["most_similar_k"])
            config["most_similar_inlist_n"] = int(config["most_similar_inlist_n"])
            config["show_m_most_similar"] = int(config["show_m_most_similar"])
            
    if config["window_strategy"] not in ["count", "time"]:
        raise ValueError("Expected sliding window strategy, pls choose between [\"count\", \"time\"]")
    elif config["window_strategy"] == "count":
        try:
            config["window_count"] = int(config["window_count"])
        except ValueError:
            raise ValueError("Expected window_count value.")
    elif config["window_strategy"] == "time":
        try:
            config["window_time"] = int(config["window_time"])
        except ValueError:
            raise ValueError("Expected window_count value.")
    
    if config["output_format"] not in ["db", "json", "parquet"]:
        raise ValueError("output_format must be one of ['db', 'json', 'parquet']")

    return config 

def check_config_validity(config):
    #### Set default values
    if config["task"] in POSSIBLE_TASKS:
        config = return_default_values(config)
    else:
        raise ValueError("Task {} not supported.".format(config["task"]))
    
    if "experiment" in config["task"]:
        #### requirements for experiments
        for path in [config["similarity_file"], config["match_file"]]:
            if path == "" or (path != "" and not os.path.exists(path)):
                raise IOError(" file {} not found. ".format(path))
        if ("log_path" not in config.keys()) or not os.path.exists(path) or path == "":
            config['log_path'] = "pipeline/logging"
        config["output_format"] = pathlib.Path(config["similarity_file"]).suffix[1:]
        return config
    
    if "smatch" in config["task"]:
        try:
            _check_stream_configuration(config)
        except Exception as e:
            print(f"[ERROR] Can not upload configuration: {e}")
        
    # try:
    #     config["ntop"] = int(config["ntop"])
    # except ValueError:
    #     raise ValueError("Expected integer ntop value.")
    # if not config["ntop"] > 0:
    #     raise ValueError("Number of neighbors to be chosen must be > 0.")

    # try:
    #     config["ncand"] = int(config["ncand"])
    # except ValueError:
    #     raise ValueError("Expected integer ncand value.")
    # if not 0 < config["ncand"] <= config["ntop"]:
    #     raise ValueError("Number of candidates must be between 0 and n_top.")

    try:
        config["sampling_factor"] = float(config["sampling_factor"])
    except ValueError:
        raise ValueError("Expected real sampling_factor value.")
    if not 1 > config["sampling_factor"] >= 0:
        raise ValueError("Sampling factor must be in [0,1).")

    if config["training_algorithm"] not in ["word2vec", "fasttext"]:
        raise ValueError(
            "Unknown training algorithm {}.".format(config["training_algorithm"])
        )
    if config["learning_method"] not in ["skipgram", "CBOW"]:
        raise ValueError("Unknown learning method {}".format(config["learning_method"]))
    for key in [
        "backtrack",
        "write_walks",
        "write_edges",
        "directed",
        "rw_stat"
    ]:
        config = _convert_to_bool(config, key)

    if "epsilon" in config:
        try:
            config["epsilon"] = float(config["epsilon"])
        except ValueError:
            print("Epsilon must be a float.")
            raise ValueError

    # if "flatten" in config:
    #     try:
    #         _convert_to_bool(config, "flatten")
    #     except ValueError:
    #         pass

    #### Path checks

    ###### WARNINGS
    if int(config["n_dimensions"]) != 300:
        warnings.warn(
            "Number of dimensions different from default (300): {}".format(
                config["n_dimensions"]
            )
        )
    if int(config["window_size"]) != 5:
        warnings.warn(
            "Window size different from default (5): {}".format(config["window_size"])
        )

    return config

def convert_token_value(original_value):
    """
    Convert a cell value into a clean string. Try to evaluate literals using ast.literal_eval first
    - If it's NaN, ***None***

    - If it's list, determine elements in the list are numeric or not
    - If it's dict, for each attribute, concat and reform as a string
    - If it's a float, round to int and convert to string
    - Otherwise, just ***str()*** the value
    - then put all values into a list except None

    : return1: list of value or None
    : return2: bool (true: numeric; false: str)

    Modify this function if we need to treat other data types
    """
    if original_value in ("", None):
        return None, False

    try:
        # Try to safely evaluate a literal value (e.g., "123", "[1, 2]", etc.)
        cell_value = ast.literal_eval(str(original_value))

        # Handle float or int
        if isinstance(cell_value, (int, float)):
            if isinstance(cell_value, float) and math.isnan(cell_value):
                return None, False
            return [str(int(cell_value))], True
        # Handle list
        elif isinstance(cell_value, list):
            if cell_value:
                is_numeric = all(isinstance(el, (int, float)) for el in cell_value)
            else: 
                is_numeric = False
            return [clean_str(str(el)) for el in cell_value], is_numeric
        # Handle dict
        elif isinstance(cell_value, dict):
            return [f"{clean_str(str(key))}_{clean_str(str(value))}" for key, value in cell_value.items()], False
        # other object (like list, dict, string)
        return [clean_str(str(cell_value))], False

    except (ValueError, SyntaxError, OverflowError):
        return [clean_str(str(original_value))], False
    
def clean_str(value):
    translation_table = str.maketrans({
        '"': r'\"',
        ',': '',
        ' ': '_'
    })
    return value.translate(translation_table)