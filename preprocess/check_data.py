import json
import pickle
from tqdm import tqdm
import os
import gc
import pandas as pd
from collections import Counter


def merge_query_file(query_file_dict_list):
    """
    The query file list is a list of dictionary of the train/validation/test queries that are separately sampled
    """
    merged_dict = {}

    for query_file_dict in tqdm(query_file_dict_list):
        for query_type in query_file_dict.keys():
            if query_type in merged_dict:
                for query, answer_dict in query_file_dict[query_type].items():
                    merged_dict[query_type][query] = answer_dict
            else:
                merged_dict[query_type] = {}
                for query, answer_dict in query_file_dict[query_type].items():
                    merged_dict[query_type][query] = answer_dict

    print({k: len(v) for k, v in merged_dict.items()})
    return merged_dict


files_dir = "/home/ec2-user/quic-efs/user/jbai/ExploreKGReasoning/input_files/"

files_to_merge = ["FB15k-betae_train_queries_part_0_29_types_big.json",
                  "FB15k-betae_train_queries_part_1_29_types_big.json",
                  "FB15k-betae_train_queries_part_2_29_types_big.json"]


dict_list = []

for file_name in files_to_merge:
    # Add iterative 1p queries
    with open(files_dir + file_name, "r") as fin:
        data_dict = json.load(fin)
        dict_list.append(data_dict)

train_data_dict_big = merge_query_file(dict_list)

filehandler = open(files_dir + "FB15k-betae_train_queries_29_types_big.json", "w")
json.dump(train_data_dict_big, filehandler)
filehandler.close()
