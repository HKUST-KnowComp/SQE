import json
import pickle
from tqdm import  tqdm
import os
import gc


dir_list = ["../sampled_data_58_eval"]
output_dir = "/home/data/jbai/input_files_58_evaluation/"

for directory_name in dir_list:

    data_names = ["FB15k-237-betae", "NELL-betae", "FB15k-betae"]

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

        for data_prefix in [data_name + "_valid_queries", data_name + "_test_queries"]:

            dict_list = []
            data_dict = {}
            for file in tqdm(all_files):
                if data_prefix in file:
                    with open(sample_data_path + file, "r") as fin:
                        data_dict = json.load(fin)
                        dict_list.append(data_dict)
            if len(dict_list) > 0:
                data_dict = merge_query_file(dict_list)

                filehandler = open(output_dir + data_prefix + "_58_types.json", "w")
                json.dump(data_dict, filehandler)
                filehandler.close()

            gc.collect()

