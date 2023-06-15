import networkx as nx
from random import sample, choice, random, randint
import numpy as np
import sys

from random import sample, choice, random, randint

import sys
from tqdm import tqdm

import math
import json
import pickle
from collections import Counter
from multiprocessing import Pool

from sample import *
import json
import pandas as pd


n_queries_train_dict_big = {
    "FB15k-betae": 273710 * 5,
    "FB15k-237-betae": 149689 * 5,
    "NELL-betae": 107982 * 5
}

n_queries_train_dict_same = {
    "FB15k-betae": 273710,
    "FB15k-237-betae": 149689,
    "NELL-betae": 107982
}


n_queries_train_dict_small = {
    "FB15k-betae": 273,
    "FB15k-237-betae": 149,
    "NELL-betae": 107
}


n_queries_train_dict_tmp = {
    "FB15k-betae": 20,
    "FB15k-237-betae": 20,
    "NELL-betae": 20
}

n_queries_valid_test_dict_same = {
    "FB15k-betae": 8000,
    "FB15k-237-betae": 5000,
    "NELL-betae": 4000
}

n_queries_valid_test_dict_tmp = {
    "FB15k-betae": 20,
    "FB15k-237-betae": 20,
    "NELL-betae": 20
}

n_queries_valid_test_dict_larger = {
    "FB15k-betae": 12000,
    "FB15k-237-betae": 7500,
    "NELL-betae": 6000
}

n_queries_valid_test_dict_larger_temp = {
    "FB15k-betae": 2,
    "FB15k-237-betae": 2,
    "NELL-betae": 2
}



if __name__ == '__main__':
    n_queries_train_dict = n_queries_train_dict_tmp
    n_queries_valid_test_dict = n_queries_valid_test_dict_tmp

    first_round_query_types = {
        "1p": "(p,(e))",
    }

    for data_dir in n_queries_train_dict.keys():

        print("Load Train Graph " + data_dir)
        train_path = "../KG_data/" + data_dir + "/train.txt"
        train_graph = GraphConstructor(train_path)
        print("nodes:", train_graph.number_of_nodes())
        print("edges:", train_graph.number_of_edges())

        print("Load Valid Graph " + data_dir)
        valid_path = ["../KG_data/" + data_dir + "/train.txt",
                      "../KG_data/" + data_dir + "/valid.txt"]

        valid_graph = GraphConstructor(valid_path)
        print("number of nodes: ", len(valid_graph.nodes))
        print("number of head-tail pairs: ", len(valid_graph.edges))

        print("Load Test Graph " + data_dir)
        test_path = ["../KG_data/" + data_dir + "/train.txt",
                     "../KG_data/" + data_dir + "/valid.txt",
                     "../KG_data/" + data_dir + "/test.txt"]
        test_graph = GraphConstructor(test_path)
        print("number of nodes: ", len(test_graph.nodes))
        print("number of head-tail pairs: ", len(test_graph.edges))

        total_edge_counter = 0
        edge_types = set()
        for u_node, v_node, attribute in train_graph.edges(data=True):
            total_edge_counter += len(attribute)
            for key in attribute.keys():
                edge_types.add(key)
        print("number of edges: ", total_edge_counter)
        print("number of relations: ", len(edge_types))

        train_graph_sampler = GraphSampler(train_graph)
        valid_graph_sampler = GraphSampler(valid_graph)
        test_graph_sampler = GraphSampler(test_graph)

        print("sample training queries")

        train_queries = {}

        def sample_train_graph_with_pattern(pattern):
            while True:

                sampled_train_query = train_graph_sampler.sample_with_pattern(pattern)

                train_query_train_answers = train_graph_sampler.query_search_answer(sampled_train_query)
                if len(train_query_train_answers) > 0:
                    break
            return sampled_train_query, train_query_train_answers

        def sample_valid_graph_with_pattern(pattern):
            while True:

                sampled_valid_query = valid_graph_sampler.sample_with_pattern(pattern)

                valid_query_train_answers = train_graph_sampler.query_search_answer(sampled_valid_query)
                valid_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_valid_query)

                if len(valid_query_train_answers) > 0 and len(valid_query_valid_answers) > 0 \
                        and len(valid_query_train_answers) != len(valid_query_valid_answers):
                    break

            return sampled_valid_query, valid_query_train_answers, valid_query_valid_answers


        def sample_test_graph_with_pattern(pattern):
            while True:

                sampled_test_query = test_graph_sampler.sample_with_pattern(pattern)

                test_query_train_answers = train_graph_sampler.query_search_answer(sampled_test_query)
                test_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_test_query)
                test_query_test_answers = test_graph_sampler.query_search_answer(sampled_test_query)


                if len(test_query_train_answers) > 0 and len(test_query_valid_answers) > 0 \
                        and len(test_query_test_answers) > 0 and len(test_query_test_answers) != len(
                    test_query_valid_answers):
                    break
            return sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers



        this_type_train_queries = {}
        one_hop_query_list = train_graph_sampler.generate_one_p_queries()
        for one_hop_query in one_hop_query_list:
            train_one_hop_query_train_answers = train_graph_sampler.query_search_answer(one_hop_query)
            if len(train_one_hop_query_train_answers) > 0:
                this_type_train_queries[one_hop_query] = {"train_answers": train_one_hop_query_train_answers}

        train_queries["(p,(e))"] = this_type_train_queries

        with open(data_dir + "_train_queries.json", "w") as file_handle:
            json.dump(train_queries, file_handle)

