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

n_queries_train_dict_big = {
    "FB15k-betae": 273710 * 3,
    "FB15k-237-betae": 149689 * 3,
    "NELL-betae": 107982 * 3
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


def GraphConstructor(file_name):
    edges = []
    graph = nx.DiGraph()

    if isinstance(file_name, list):
        for _file_name in file_name:
            with open(_file_name, "r") as file_in:
                for line in file_in:
                    line_list = line.strip().split("\t")

                    line_numerical_list = [line_list[0], line_list[2], {line_list[1]: 1}]
                    edges.append(line_numerical_list)

    else:
        with open(file_name, "r") as file_in:
            for line in file_in:
                line_list = line.strip().split("\t")

                line_numerical_list = [line_list[0], line_list[2], {line_list[1]: 1}]
                edges.append(line_numerical_list)

    graph.add_edges_from(edges)
    return graph


# The class that is used for sampling from a networkx graph.
class GraphSampler:
    def __init__(self, graph):

        self.graph = graph
        self.dense_nodes = list(graph.nodes)


    def iterative_sample_with_pattern(self, pattern="(p,(e))"):

        result_query_list = []
        for node in tqdm(self.dense_nodes):
            _query, _ = _pattern_to_query(pattern, self.graph, node)
            if _query is not None:
                result_query_list.append(_query)

        return result_query_list

    def generate_one_p_queries(self):
        result_query_list = []
        for node in tqdm(self.dense_nodes):
            for tail_node, attribute_dict in self.graph[node].items():
                # "(p,(40),(e,(2429)))"
                for key in attribute_dict.keys():
                    result_query_list.append("(p,(" + str(key) + "),(e,(" + str(node) + ")))")

        return list(set(result_query_list))

    # The function used to call the recursion of sampling queries from the ASER graph.
    def sample_with_pattern(self, pattern):
        while True:
            random_node = sample(self.dense_nodes, 1)[0]
            _query, _ = _pattern_to_query(pattern, self.graph, random_node)
            if _query is not None:
                return _query

    # The function used for finding the answers to a query
    def query_search_answer(self, query):

        graph = self.graph

        query = query[1:-1]
        parenthesis_count = 0

        sub_queries = []
        jj = 0

        for ii, character in enumerate(query):
            # Skip the comma inside a parenthesis
            if character == "(":
                parenthesis_count += 1

            elif character == ")":
                parenthesis_count -= 1

            if parenthesis_count > 0:
                continue

            if character == ",":
                sub_queries.append(query[jj: ii])
                jj = ii + 1

        sub_queries.append(query[jj: len(query)])

        if sub_queries[0] == "p":

            sub_query_answers = self.query_search_answer(sub_queries[2])

            all_answers = []

            for answer_id, sub_answer in enumerate(sub_query_answers):
                next_nodes = list(graph.neighbors(sub_answer))

                for node in next_nodes:
                    if sub_queries[1][1:-1] in graph.edges[sub_answer, node]:
                        all_answers.append(node)
            all_answers = list(set(all_answers))
            return all_answers

        elif sub_queries[0] == "e":
            return [sub_queries[1][1:-1]]

        elif sub_queries[0] == "i":

            sub_query_answers_list = []

            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.query_search_answer(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers & set(sub_query_answers)

            merged_answers = list(merged_answers)

            return merged_answers

        elif sub_queries[0] == "u":

            sub_query_answers_list = []
            for i in range(1, len(sub_queries)):
                sub_query_answers_i = self.query_search_answer(sub_queries[i])
                sub_query_answers_list.append(sub_query_answers_i)

            merged_answers = set(sub_query_answers_list[0])
            for sub_query_answers in sub_query_answers_list:
                merged_answers = merged_answers | set(sub_query_answers)

            merged_answers = list(merged_answers)

            return merged_answers
        elif sub_queries[0] == "n":
            sub_query_answers = self.query_search_answer(sub_queries[1])
            all_nodes = list(self.graph.nodes)
            negative_answers = [node for node in all_nodes if node not in sub_query_answers]

            negative_answers = list(set(negative_answers))
            return negative_answers

        else:
            print("Invalid Pattern")
            exit()

    # The function used for finding a query that have at least one answer
    def sample_valid_question_with_answers(self, pattern):
        while True:
            _query = self.sample_with_pattern(pattern)
            _answers = self.query_search_answer(_query)
            if len(_answers) > 0:
                return _query, _answers


def _pattern_to_query(pattern, graph, node):
    """
    In this function, _pattern_to_query is recursively used for finding the anchor nodes and relations from
    a randomly sampled entity, which is assumed to be the answer.

    :param pattern:
    :param graph:
    :param node:
    :return:
    """

    pattern = pattern[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    for ii, character in enumerate(pattern):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(pattern[jj: ii])
            jj = ii + 1

    sub_queries.append(pattern[jj: len(pattern)])

    if sub_queries[0] == "p":

        reversely_connected_nodes = np.array([next_node for next_node in list(graph.predecessors(node))])
        if len(reversely_connected_nodes) == 0:
            return None, None

        next_node = choice(reversely_connected_nodes)

        # min_edge_value = min(list(graph.edges[next_node, node]["total_score"].values()))

        edge_name = choice([k for k in graph.edges[next_node, node].keys()])

        sub_query, _ = _pattern_to_query(sub_queries[1], graph, next_node)
        if sub_query is None:
            return None, None

        return "(p,(" + edge_name + '),' + sub_query + ")", next_node

    elif sub_queries[0] == "n":
        """If we use the negation here, it is possible that we generate a query that do not have an answer.
        But the overall chance is small. Anyway, when we cannot find an answer we just sample again.
        
        After modification, we choose to use the same node for sampling to enable the negation query do have an effect
        on the final outcome
        """

        # random_node = sample(list(graph.nodes()), 1)[0]
        sub_query, returned_node = _pattern_to_query(sub_queries[1], graph, node)
        if sub_query is None:
            return None, None

        return "(n," + sub_query + ")", returned_node

    elif sub_queries[0] == "e":
        return "(e,(" + node + "))", str(node)

    elif sub_queries[0] == "i":

        sub_queries_list = []

        next_node_list = []

        for i in range(1, len(sub_queries)):
            sub_q, _next_node = _pattern_to_query(sub_queries[i], graph, node)
            sub_queries_list.append(sub_q)

            next_node_list.append(_next_node)

        for sub_query in sub_queries_list:
            if sub_query is None:
                return None, None

        for index_i, sub_query_i in enumerate(sub_queries_list):
            for index_j in range(index_i + 1, len(sub_queries_list)):
                if sub_query_i == sub_queries_list[index_j]:
                    return None, None

                if next_node_list[index_i] == next_node_list[index_j]:
                    return None, None

        return_str = "(i"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return return_str, node

    elif sub_queries[0] == "u":
        # randomly sample a node
        sub_queries_list = []
        next_node_list = []

        random_subquery_index = randint(1, len(sub_queries) - 1)

        # The answer only need to be one of the queries
        for i in range(1, len(sub_queries)):
            if i == random_subquery_index:
                sub_q, _next_node = _pattern_to_query(sub_queries[i], graph, node)
            else:
                sub_q, _next_node = _pattern_to_query(sub_queries[i], graph, sample(list(graph.nodes()), 1)[0])

            if sub_q is None:
                return None, None

            sub_queries_list.append(sub_q)
            next_node_list.append(_next_node)

        if len(sub_queries_list) == 0:
            return None, None

        return_str = "(u"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return return_str, node

    else:
        print("Invalid Pattern")
        exit()




if __name__ == '__main__':
    n_queries_train_dict = n_queries_train_dict_tmp
    n_queries_valid_test_dict = n_queries_valid_test_dict_tmp



    first_round_query_types = {
        "2p": "(p,(p,(e)))",
        "3p": "(p,(p,(p,(e))))",
        "2i": "(i,(p,(e)),(p,(e)))",
        "3i": "(i,(p,(e)),(p,(e)),(p,(e)))",
        "ip": "(p,(i,(p,(e)),(p,(e))))",
        "pi": "(i,(p,(p,(e))),(p,(e)))",
        "2u": "(u,(p,(e)),(p,(e)))",
        "up": "(p,(u,(p,(e)),(p,(e))))",
        "2in": "(i,(n,(p,(e))),(p,(e)))",
        "3in": "(i,(n,(p,(e))),(p,(e)),(p,(e)))",
        "inp": "(p,(i,(n,(p,(e))),(p,(e))))",
        "pin": "(i,(p,(p,(e))),(n,(p,(e))))",
        "pni": "(i,(n,(p,(p,(e)))),(p,(e)))",
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



        for query_type, sample_pattern in first_round_query_types.items():
            print("train query_type: ", query_type)

            this_type_train_queries = {}

            if "n" in query_type:
                n_query = n_queries_train_dict[data_dir] // 10
            else:
                n_query = n_queries_train_dict[data_dir]

            pattern_list = []
            for _ in tqdm(range(n_query)):
                # if len(pattern_list) < num_processes:
                #     pattern_list.append(sample_pattern)
                #
                # else:
                #     with Pool(num_processes) as p:
                #         answer_tuple_list = p.map(sample_train_graph_with_pattern, pattern_list)
                #         for sampled_train_query, train_query_train_answers in answer_tuple_list:
                #             this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}
                #     pattern_list = []

                sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern(sample_pattern)
            # with Pool(num_processes) as p:
            #     answer_tuple_list = p.map(sample_train_graph_with_pattern, pattern_list)
            #     for sampled_train_query, train_query_train_answers in answer_tuple_list:
                this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}



            train_queries[sample_pattern] = this_type_train_queries

        this_type_train_queries = {}
        one_hop_query_list = train_graph_sampler.iterative_sample_with_pattern()
        for one_hop_query in one_hop_query_list:
            train_one_hop_query_train_answers = train_graph_sampler.query_search_answer(one_hop_query)
            if len(train_one_hop_query_train_answers) > 0:
                this_type_train_queries[one_hop_query] = {"train_answers": train_one_hop_query_train_answers}

        train_queries["(p,(e))"] = this_type_train_queries

        with open(data_dir + "_train_queries.json", "w") as file_handle:
            json.dump(train_queries, file_handle)

        print("sample validation queries")

        validation_queries = {}
        for query_type, sample_pattern in first_round_query_types.items():
            print("validation query_type: ", query_type)

            this_type_validation_queries = {}

            n_query = n_queries_valid_test_dict[data_dir]

            for _ in tqdm(range(n_query)):

                sampled_valid_query, valid_query_train_answers, valid_query_valid_answers = \
                    sample_valid_graph_with_pattern(sample_pattern)

                this_type_validation_queries[sampled_valid_query] = {
                    "train_answers": valid_query_train_answers,
                    "valid_answers": valid_query_valid_answers
                }
                # print(len(valid_query_train_answers))
                # print(len(valid_query_valid_answers))
            validation_queries[sample_pattern] = this_type_validation_queries

        with open(data_dir + "_valid_queries.json", "w") as file_handle:
            json.dump(validation_queries, file_handle)

        print("sample testing queries")

        test_queries = {}

        for query_type, sample_pattern in first_round_query_types.items():
            print("test query_type: ", query_type)
            this_type_test_queries = {}

            n_query = n_queries_valid_test_dict[data_dir]

            for _ in tqdm(range(n_query)):

                sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers = \
                    sample_test_graph_with_pattern(sample_pattern)


                this_type_test_queries[sampled_test_query] = {
                    "train_answers": test_query_train_answers,
                    "valid_answers": test_query_valid_answers,
                    "test_answers": test_query_test_answers
                }

            test_queries[sample_pattern] = this_type_test_queries
        with open(data_dir + "_test_queries.json", "w") as file_handle:
            json.dump(test_queries, file_handle)