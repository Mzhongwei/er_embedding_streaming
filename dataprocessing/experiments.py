import json
import logging
from pathlib import Path
import sqlite3
import math
import pandas as pd
from utils.write_log import write_log
from scipy.stats import kendalltau

from dataprocessing.similarity_anlysis import similarity_analysis

def _get_ground_truth(ground_truth_file):
    matches = {}
    n_lines = 0
    with open(ground_truth_file, 'r', encoding='utf-8') as fp:
        for n, line in enumerate(fp.readlines()):
            if len(line.strip()) > 0:
                item, match = line.replace('_', '__').split(',')
                if item not in matches:
                    matches[item] = [match.strip()]
                else:
                    matches[item].append(match.strip())
                n_lines = n
        if n_lines == 0:
            raise IOError('Matches file is empty. ')
    return matches


def _get_similarity_list(similarity_file, output_format):
    sim_list = {}
    if output_format == "db":
        try:
            if Path(similarity_file).exists:
                conn = sqlite3.connect(similarity_file)
                cursor = conn.cursor()

                table = "matchinglist"
                primary_key = "id"
                value = "similarity"

                cursor.execute(f'SELECT {primary_key}, {value} FROM {table}')
                rows = cursor.fetchall()

                sim_list = {row[0]: json.loads(row[1]) for row in rows}
                conn.close()
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    elif output_format == "json":
        print("parsing json file...")
        try:
            if Path(similarity_file).exists:
                with open(similarity_file, "r", encoding="utf-8") as f:
                    sim_list = json.load(f)
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    elif output_format == "parquet":
        try:
            if Path(similarity_file).exists:
                data = pd.read_parquet(similarity_file)
                sim_list = dict(zip(data['key'], data['value'].apply(json.loads)))
            else:
                raise ValueError("[ERROR] Do not find similarity file.")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
    return sim_list


def _get_match_pairs(data_dict):
    '''
    :param item: {'key': ['match1', 'match2'], 'key2': ['match1', 'match2']}
    :return: ex. (item, match1), (item, match2)
    '''
    matchpair_set = set()
    for key in data_dict:
        item_matched = int(key.split('__')[1])
        for value in data_dict[key]:
            item_matching = None
            # value ex. ['idx__27463', 'idx__19499', 'idx__18194', 'idx__14572', 'idx__64547', 'idx__35496', 'idx__22535']
            if isinstance(value, str):
                item_matching = value
            item_matching = int(item_matching.split('__')[1])

            # sort: facilitate deplication
            if item_matching > item_matched:
                el = (item_matched, item_matching)
            else:
                el = (item_matching, item_matched)
            matchpair_set.add(el)
    return matchpair_set

def _get_similar_pairs(data_dict, n, appr):
    '''
    :param data_dict: {'key': [(sim_degree, 'match1'), (sim_degree, 'match2')], 'key2': [(sim_degree, 'match1'), (sim_degree, 'match2')]}
    :param n: select top n similarities
    :param appr: number of decimal places to be retained for similarity
    :return: ex. (item, match1), (item, match2)
    '''
    matchpair_set = set()
    for key in data_dict:
        item_matched = int(key.split('__')[1])
        values = [[round(degree, appr), _] for degree, _ in data_dict[key]]
        if isinstance(n, int) and n>0:
            # find the n most matching scores
            scores = list(set(degree for degree, _ in values))
            scores = sorted(scores, reverse=True)[:n]
            values = [value for value in values if value[0] >= scores[len(scores)-1]]
        for item in values:
            item_matching = None
            # value ex. [[0.8194172382354736, 'idx__38260'], [0.8130440711975098, 'idx__51652']]
            if isinstance(item, list):
                item_matching = item[1]
                print(item_matching)
                item_matching = int(item_matching.split('__')[1])

            # sort: facilitate deplication
            if item_matching > item_matched:
                el = (item_matched, item_matching)
            else:
                el = (item_matching, item_matched)
            matchpair_set.add(el)
    return matchpair_set

def _get_similarity_pairs_with_degree(data_dict, n, appr):
    sim_list = {}
    for key in data_dict:
        item_matched = int(key.split('__')[1])
        values = [[round(degree, appr), _] for degree, _ in data_dict[key]]
        if isinstance(n, int) and n>0:
            # find the n most matching scores
            scores = list(set(degree for degree, _ in values))
            scores = sorted(scores, reverse=True)[:n]
            values = [value for value in values if value[0] >= scores[len(scores)-1]]
            sim_list[item_matched] = values
    return sim_list


def compare_ground_truth(configuration):
    """
    Test the accuracy of matches by precisin, recall, f1
    :param configuration:
    """
    ground_truth_file = configuration['match_file']
    similarity_file = configuration['similarity_file']
    output_format = configuration['output_format']
    # similarity_list example: {item: [matches]}
    similarity_list = _get_similarity_list(similarity_file, output_format)
    
    # matches example: {item: [matches]}
    matches = _get_ground_truth(ground_truth_file)

    correct_matches = 0
    predicted_matches = _get_similar_pairs(similarity_list, int(configuration['n_first']), int(configuration['approximate']))
    actual_matches = _get_match_pairs(matches)

    total_predicted_matches = len(predicted_matches)
    print(total_predicted_matches)
    total_relevant_matches = len(actual_matches)
    print(total_relevant_matches)
    correct_matches =  len(set(predicted_matches) & set(actual_matches)) 
    print(correct_matches)

    if configuration['sim_vis'] == "true":
        sim_list = _get_similarity_pairs_with_degree(similarity_list, int(configuration['n_first']), int(configuration['approximate']))
        similarity_analysis(sim_list, actual_matches, configuration['output_file_name'])
    # Precision: Number of correct matches / Total predicted matches
    precision = correct_matches / total_predicted_matches if total_predicted_matches != 0 else 0.0
    # Recall: Number of correct matches / Total relevant matches
    recall = correct_matches / total_relevant_matches if total_relevant_matches != 0 else 0.0
    f1_score = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0.0

    # print results
    print(f'''Experiment result for {similarity_file}: \n correct matches: {correct_matches} \n total number of predicted matches: {total_predicted_matches} \n total number of matches in groud truth file: {total_relevant_matches} \n \n precision: {precision} \n recall: {recall} \n f1 score: {f1_score}''')

    # output results to log file
    dir_name = "experiments"
    Path(f'''{configuration["log_path"]}/{dir_name}''').mkdir(parents=True, exist_ok=True)
    logger = write_log(f'''{configuration["log_path"]}''', dir_name, dir_name)
    
    logger.info(f'''[RESULTS] Experiment result of similarity list in file [{similarity_file}] by taking records with top {configuration["n_first"]} similarity and the similarity retains {configuration['approximate']} decimal places: \n correct matches: {correct_matches} \n total number of predicted matches: {total_predicted_matches} \n total number of matches in groud truth file: {total_relevant_matches} \n \n precision: {precision} \n recall: {recall} \n f1 score: {f1_score}''')
    
    # return precision, recall, f1_score

def __get_dict_to_compare(input_list, configuration):
    output_dict = {}
    for el in input_list:
        id = el.split('__')[1]
        if int(id) > int(configuration["source_num"]):
            output_dict[el]= input_list[el]
    return output_dict

def __compare_kendall(configuration):
    similarity_list = _get_similarity_list(configuration['similarity_file'], "db")
    similarity_list_ken = _get_similarity_list(configuration['similarity_file_ken'], "db")

    sim_to_compare = __get_dict_to_compare(similarity_list, configuration)
    sim_to_compare_ken = __get_dict_to_compare(similarity_list_ken, configuration)

    # Intersection Matching
    num_nan = 0
    num_tau = 0
    total_tau = 0
    p_value_ok = 0
    positive = 0
    for key in sim_to_compare:
        if key in sim_to_compare_ken:
            
            # exact values of one common key for each list
            sim_id = [x[1] for x in sim_to_compare[key]]
            sim_id_ken = [x[1] for x in similarity_list_ken[key]]

            # get common values of these two lists
            common_labels = set(sim_id) & set(sim_id_ken)

            # filter these values with their original order
            sim_filtered = [x for x in sim_id if x in common_labels]
            sim_filtered_ken = [x for x in sim_id_ken if x in common_labels]

            # build a map
            sim_map = {label: rank for rank, label in enumerate(sim_filtered)}
            sim_map_ken = {label: rank for rank, label in enumerate(sim_filtered_ken)}
            
            # get a rank
            sim_rank = [sim_map[label] for label in sim_filtered]
            sim_rank_ken = [sim_map_ken[label] for label in sim_filtered]

            # calculate Kendall's Tau
            tau, p_value = kendalltau(sim_rank, sim_rank_ken)

            if math.isnan(float(tau)):
                num_nan = num_nan + 1
            else:
                num_tau = num_tau + 1
                total_tau += tau
                if p_value < 0.5 and math.isnan(float(p_value)) == False:
                    p_value_ok = p_value_ok + 1
                    if tau > 0:
                        positive = positive + 1
                    
                    print(key)
                    print(sim_map)
                    print(sim_map_ken)
                    print(f"tau: {tau} \n p_value: {p_value}")
    average_tau = total_tau/num_tau
            
    print(f"average tau: {average_tau}, number of nan : {num_nan}, p_value : {p_value_ok}, positive: {positive}")
    # Path(f'''{configuration["log_path"]}/experiments''').mkdir(parents=True, exist_ok=True)
    # logger = write_app_log(f'''{configuration["log_path"]}/experiments/''')
    
    # logger.info(f"Kendall's Tau for file {configuration['similarity_file']} \n tau: {tau} \n p_value: {p_value}")
    