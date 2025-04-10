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

def _return_default_values_exp(config):
    default_values = {
        "n_first": 3,
        "approximate": 16,
        "sim_vis": True,
        "source_num":0
    }

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]

    config["n_first"] = int(config["n_first"])
    config["approximate"] = int(config["approximate"])
    config["source_num"] = int(config["source_num"])

    config = _convert_to_bool(config, ["sim_vis"])

    return config

def _return_default_values_gwe(config):
    default_values = {
        "node_types":["7#__tn", "7$__tt", "3$__idx", "1$__cid"],
        "flatten": "no",
        "smoothing_method": "no",
        "directed":False,
        "write_walks": True,
        "walk_length": 60,
        "walks_number": 20,
        "backtrack": False,
        "rw_stat":True,
        "learning_method": "skipgram",
        "window_size": 3,
        "n_dimensions": 300,
        "training_algorithm": "word2vec",
        "sampling_factor": 0.001,
    }

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    
    config['node_types'] = ast.literal_eval(config['node_types'])

    config["walk_length"] = int(config["walk_length"])
    config["walks_number"] = int(config["walks_number"])
    config["window_size"] = int(config["window_size"])
    config["n_dimensions"] = int(config["n_dimensions"])

    if config["smoothing_method"] not in ["log", "no"]:
        raise ValueError("Unknown smoothing_method {}".format(config["smoothing_method"]))

    if config["training_algorithm"] not in ["word2vec", "fasttext"]:
        raise ValueError(
            "Unknown training algorithm {}.".format(config["training_algorithm"])
        )
    if config["learning_method"] not in ["skipgram", "CBOW"]:
        raise ValueError("Unknown learning method {}".format(config["learning_method"]))
    
    for key in [
        "directed",
        "write_walks",
        "backtrack",      
        "rw_stat"
    ]:
        config = _convert_to_bool(config, key)

    return config
                
def _return_default_values_sk(config):
    default_values = {
        "top_k":10,
        "simlist_n":10,
        "simlist_show":5,
        "strategy_suppl":"faiss",
        "output_format":"db",
        "source_num":0,
        "kafka_topicid":"entity_resolution_process",
        "kafka_groupid":"er_group",
        "bootstrap_servers":"localhost",
        "port":"9092",
        "window_strategy":"count",
        "window_count":136,
        "update_frequency":0
    }

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]

    config["update_frequency"] = int(config["update_frequency"])
    config["top_k"] = int(config["top_k"])
    config["simlist_n"] = int(config["simlist_n"])
    config["simlist_show"] = int(config["simlist_show"])
    config["source_num"] = int(config["source_num"])
            
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
    if config["strategy_suppl"] not in ["faiss", "basic"]:
        raise ValueError('''strategy_suppl must be one of ["faiss", "basic"]''')
    
    return config

def _check_file(config, files):
    for file in files:
        if file not in config:
            raise ValueError(f"pls specify the file {file}")
        else:
            file_path = config[file]
            if file_path == "" or (file_path != "" and not os.path.exists(file_path)):
                raise IOError("File {} not found. ".format(file_path))
            
def check_config_validity(config):
    #### Set default values
    if config["task"] not in POSSIBLE_TASKS:
        raise ValueError("Task {} not supported.".format(config["task"]))
 
    if "experiment" in config["task"]:
        config = _return_default_values_exp(config)
        _check_file(config, ['similarity_file', 'match_file'])
        config["output_format"] = pathlib.Path(config["similarity_file"]).suffix[1:]
    elif "smatch" in config["task"]:
        config = _return_default_values_gwe(config)
        config = _return_default_values_sk(config)
    elif "batch" in config["task"]:
        config = _return_default_values_gwe(config)
        _check_file(config, ['dataset_file'])

    if ("log_path" not in config.keys()) or not os.path.exists(config['log_path']) or config['log_path'] == "":
        config['log_path'] = "pipeline/logging" 
    if "output_file_name" not in config:
        config["output_file_name"] = "test"

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