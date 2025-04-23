import argparse
import ast
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML

import warnings

from gensim.models import FastText, Word2Vec



with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from utils.utils import *
    from utils.write_log import write_log
    from dataprocessing.evaluation import compare_ground_truth
    from dynamic_embedding.dynamic_graph import dyn_graph_generation
    from dynamic_embedding.dynamic_embeddings import initialize_embeddings
    from dynamic_embedding.dynamic_sampling import dynrandom_walks_generation
    from dataprocessing.kafkaconsumer import start_kafka_consumer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unblocking', action='store_true', default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--config_file', action='store', default=None)
    group.add_argument('-d', '--config_dir', action='store', default=None)
    parser.add_argument('--no_info', action='store_true', default=False)
    args = parser.parse_args()
    return args

def batch_driver(configuration):
    config_logger = write_log(configuration['log']['path'], "config", "batch")
    print(f"Saving configuration setting in log file...")
    config_logger.info(f"Configuration for batch test: {json.dumps(configuration)}")
    print('Config saved!')
    
    dataset_file = configuration['dataset_file']
    print('loading edgelist file...')

    df_table = pd.read_csv(dataset_file) 
    # Add a new column 'rid' using row index
    id_nums = len(df_table)
    df_table["rid"] = ["idx__{}".format(i) for i in range(id_nums)] 

    # generate graph and save
    graph = dyn_graph_generation(configuration)
    graph.set_id_nums(int(id_nums-1))
    print(f"graph attributes: id_nums-{graph.get_id_nums()}, smooth method-{graph.get_smooth_method()}, directed-{graph.get_directed_info()}")
    graph.build_relation(df_table)

    print(f"dyn roots len: {len(graph.dyn_roots)}")

    # random walk
    walks = dynrandom_walks_generation(configuration, graph)
    
    # training model
    embeddings_file = f"pipeline/embeddings/{configuration['output_file_name']}.emb"
    print("create a new model...")
    model = initialize_embeddings(write_walks=configuration['walks']['write_walks'],
                dimensions=configuration['embeddings']['n_dimensions'],
                window_size=configuration['embeddings']['window_size'],
                training_algorithm=configuration['embeddings']['training_algorithm'],
                learning_method=configuration['embeddings']['learning_method'],
                sampling_factor=configuration['embeddings']['sampling_factor'])
    print("start training...")
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=10)
    model.save(embeddings_file)
    
    print(f"Saving graph with attributes... Graph file path: /home/zhongwei/Data_ingestion/embIng/pipeline/graph")
    g = graph.clean_attributes()
    g.write_graphml(f"pipeline/graph/{configuration['output_file_name']}.graphml")
    print("Graph saved!")

def streaming_driver(configuration):
    '''This function initiates Graph and Embedding model for the streaming process. 
    Once the initiation finishes, driver will call kafka consumerwhich receives the data and performs the task ER

    '''
    config_logger = write_log(configuration['log']['path'], "config", "stream")
    print(f"Saving configuration setting in log file...")
    config_logger.info(f"Configuration for batch test: {json.dumps(configuration)}")
    print('Config saved!')

    ########### init #########
    ##### load edgelist
    graph_file = configuration['graph_file']
    print("walks_number stream", configuration['walks']['walks_number'])

    ##### generate empty Graph
    graph = dyn_graph_generation(configuration)
    if os.path.exists(graph_file):
        print('load graph file...')
        ##### load graph
        graph.load_graph(graph_file)
        # check configuration with graph properties
        if configuration['graph']['smoothing_method'] != graph.get_smooth_method():
            raise ValueError(f"smooth method setting in the config doesn't correspond to the graph attribute.")
        if configuration['graph']['directed'] != graph.get_directed_info():
            raise ValueError(f"[directed] value setting in the config doesn't correspond to the graph attribute.")
        print(f"graph attributes: id_nums-{graph.get_id_nums()}, smooth method-{graph.get_smooth_method()}, directed-{graph.get_directed_info()}")
        configuration["source_num"] = graph.get_id_nums()
        ##### load model
        embeddings_file = configuration['embeddings_file']
        if configuration['embeddings']['window_size'] == 'fasttext':
            print('load fasttext model...')
            model = FastText.load(embeddings_file)
        else:
            print('load word2vec model...')
            model = Word2Vec.load(embeddings_file)
    else:
        print(f"graph attributes: id_nums-{graph.get_id_nums()}, smooth method-{graph.get_smooth_method()}, directed-{graph.get_directed_info()}")
        ###### create an empty model
        print("Create a new model...")
        model = initialize_embeddings(write_walks=configuration['walks']['write_walks'],
                    dimensions=configuration['embeddings']['n_dimensions'],
                    window_size=configuration['embeddings']['window_size'],
                    training_algorithm=configuration['embeddings']['training_algorithm'],
                    learning_method=configuration['embeddings']['learning_method'],
                    sampling_factor=configuration['embeddings']['sampling_factor'])
    configuration["source_num"] = graph.get_id_nums()
    ########### stream part ##############
    print('Streaming...')
    output_file_name = configuration['output_file_name']
    start_kafka_consumer(configuration, graph, model, output_file_name)

def evaluation_driver(configuration):
    compare_ground_truth(configuration)

def read_configuration(config_file):
    # TODO: convert this to reading toml
    config = {}

    with open(config_file, 'r') as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if len(line) == 0 or line[0] == '#': continue
            split_line = line.split(':')
            if len(split_line) < 2:
                continue
            else:
                key, value = split_line
                value = value.strip()
                config[key] = value
    return config

def load_yaml_config(config_file):
    yaml = YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config


def full_run(config_dir, config_file):
    # Parsing the configuration file.
    path = Path(config_dir, config_file)
    configuration = load_yaml_config(path)
    # Checking the correctness of the configuration, setting default values for missing values.
    configuration = check_config_validity(configuration)

    # Running the task specified in the configuration file.

    if configuration['task'] == 'smatch': # smatch : stream match
        streaming_driver(configuration)
    elif configuration['task'] == 'evaluation':
        evaluation_driver(configuration)
    elif configuration['task'] == "batch":
        batch_driver(configuration)
        


def main(file_path=None, dir_path=None, args=None):
    results = None
    configuration = None

    # Building dir tree required to run the code.
    os.makedirs('pipeline/embeddings', exist_ok=True)
    os.makedirs('pipeline/logging', exist_ok=True)
    os.makedirs('pipeline/graph', exist_ok=True)
    os.makedirs('pipeline/stat', exist_ok=True)
    os.makedirs('pipeline/similarity', exist_ok=True)

    # Finding the configuration file paths.
    if args:
        if args.config_dir:
            config_dir = args.config_dir
            config_file = None
        else:
            config_dir = None
            config_file = args.config_file
        unblocking = args.unblocking
    else:
        config_dir = dir_path
        config_file = file_path
        unblocking = False

    # Extracting valid files
    if config_dir:
        config_dir = Path(config_dir)
        valid_files = []
        for f in config_dir.iterdir():
            if f.is_file() and not f.name.startswith('default'):
                valid_files.append(f.name)
        n_files = len(valid_files)
        print(f'Found {n_files} files in {config_dir}')
    elif config_file:
        config_path = Path(args.config_file if args else config_file)
        config_dir = config_path.parent
        valid_files = [config_path.name]
    else:
        raise ValueError('Missing file_path or config_path.')

    if unblocking:
        print('######## IGNORING EXCEPTIONS ########')
        for idx, file in enumerate(sorted(valid_files)):
            try:
                print('#' * 80)
                print(f'# File {idx + 1} out of {valid_files}')
                print(f'# Configuration file: {file}')
                t_start = datetime.datetime.now()
                print(OUTPUT_FORMAT.format('Starting run.', t_start.strftime(TIME_FORMAT)))
                print()

                full_run(config_dir, file)

                t_end = datetime.datetime.now()
                print(OUTPUT_FORMAT.format('Ending run.', t_end.strftime(TIME_FORMAT)))
                dt = t_end - t_start
                print('# Time required: {:.2} s'.format(dt.total_seconds()))
            except Exception as e:
                print(f'Run {file} has failed. ')
                print(e)
    else:
        for idx, file in enumerate(sorted(valid_files)):
            print('#' * 80)
            print(f'# File {idx + 1} out of {valid_files}')
            print(f'# Configuration file: {file}')
            t_start = datetime.datetime.now()
            print(OUTPUT_FORMAT.format('Starting run.', t_start.strftime(TIME_FORMAT)))
            print()

            full_run(config_dir, file)

            t_end = datetime.datetime.now()
            print(OUTPUT_FORMAT.format('Ending run.', t_end.strftime(TIME_FORMAT)))
            dt = t_end - t_start
            print('# Time required: {:.2f} s'.format(dt.total_seconds()))

if __name__ == '__main__':
    args = parse_args()
    main(args=args)
