from sample import *
import json
import pandas as pd

num_processes = 20

def sample_301_test_data(id):

    n_queries_valid_test_dict = n_queries_valid_test_dict_larger_temp
    all_query_types = pd.read_csv("../preprocess/test_generated_formula_anchor_node=3.csv").reset_index(drop = True)#debug

    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]
        original_query_types[fid]=query


    for data_dir in n_queries_valid_test_dict.keys():

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

        ###############validation sampling procedure################
        print("sample validation queries")

        validation_queries = {}
        for query_type, sample_pattern in original_query_types.items():
            print("validation query_type: ", query_type)
            
            this_type_validation_queries = {}

            n_query = n_queries_valid_test_dict[data_dir]
            if(query_type == "type0000"):
                n_query *= 2 # in total 3 times size for 1p queries
            n_query = n_query // num_processes
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

        with open("../sampled_data_301_larger/" + data_dir + "_valid_queries_" + str(id) + ".json", "w") as file_handle:
            json.dump(validation_queries, file_handle)
        ###############test sampling procedure################
        print("sample testing queries")

        test_queries = {}

        for query_type, sample_pattern in original_query_types.items():
            print("test query_type: ", query_type)
            this_type_test_queries = {}

            n_query = n_queries_valid_test_dict[data_dir]
            if(query_type == "type0000"):
                n_query *= 2 # in total 3 times size for 1p queries
            n_query = n_query // num_processes
            for _ in tqdm(range(n_query)):
                sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers = \
                    sample_test_graph_with_pattern(sample_pattern)

                this_type_test_queries[sampled_test_query] = {
                    "train_answers": test_query_train_answers,
                    "valid_answers": test_query_valid_answers,
                    "test_answers": test_query_test_answers
                }

            test_queries[sample_pattern] = this_type_test_queries
        with open("../sampled_data_301_larger/" + data_dir + "_test_queries_" + str(id) + ".json", "w") as file_handle:
            json.dump(test_queries, file_handle)

if __name__ == "__main__":
    with Pool(num_processes) as p:
        print(p.map(sample_301_test_data, range(num_processes)))