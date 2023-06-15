import json
import pickle
from tqdm import tqdm
import os
import gc
import pandas as pd
from collections import Counter

all_query_types = pd.read_csv("../preprocess/test_generated_formula_anchor_node=2.csv").reset_index(drop = True)#debug

original_query_types = {}
for i in range(all_query_types.shape[0]):
    fid = all_query_types.formula_id[i]
    query = all_query_types.original[i]
    original_query_types[fid]=query

dir_list = ["/home/data/jbai/sampled_data_29_train"]
output_dir = "../sampled_data_29_train/"

for directory_name in dir_list:

    data_names = ["FB15k-betae", "NELL-betae", "FB15k-237-betae"]

    all_files = os.listdir(directory_name)
    sample_data_path = directory_name + "/"

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


    for data_name in data_names:
        print(data_name)
        num_parts = 32

        train_data_prefix = data_name + "_train_queries"

        for part_id in range(num_parts):
            print("Part: ", part_id)

            train_dict_list_big = []
            train_data_dict_big = {}

            # Add iterative 1p queries
            with open("../sampled_1p_train/" + train_data_prefix + ".json", "r") as fin:
                data_dict = json.load(fin)
                train_dict_list_big.append(data_dict)

            file_counter = Counter()
            for file in tqdm(all_files):
                if data_name in file:
                    query_id = file.split("_")[1]
                    query_form = original_query_types[query_id]
                    if file_counter[query_form] % num_parts == part_id:
                        with open(sample_data_path + file, "r") as fin:

                            data_dict = json.load(fin)

                            data_dict = {query_form: data_dict}

                            train_dict_list_big.append(data_dict)

                    file_counter[query_form] += 1

            print("Part: ", part_id, "/", num_parts, " #big:  ", len(train_dict_list_big))

            train_data_dict_big = merge_query_file(train_dict_list_big)

            # filehandler = open(output_dir + train_data_prefix + "_part_" + str(part_id) + "_29_types_big.json", "w")

            filehandler = open(output_dir + train_data_prefix + "_" + str(part_id) + ".json", "w")

            json.dump(train_data_dict_big, filehandler)
            filehandler.close()


            gc.collect()




