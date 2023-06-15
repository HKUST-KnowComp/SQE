from sample import *
import json
import pandas as pd

num_processes = 32

def sample_301_test_data(id):

    n_queries_test_dict = n_queries_train_dict_small#[20,20,20]
    all_query_types = pd.read_csv("../preprocess/test_generated_formula_anchor_node=3.csv").reset_index(drop = True)#debug

    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]
        original_query_types[fid]=query


    for data_dir in n_queries_test_dict.keys():

        print("Load Train Graph " + data_dir)
        train_path = "../KG_data/" + data_dir + "/train.txt"
        train_graph = GraphConstructor(train_path)
        print("nodes:", train_graph.number_of_nodes())
        print("edges:", train_graph.number_of_edges())


        total_edge_counter = 0
        edge_types = set()
        for u_node, v_node, attribute in train_graph.edges(data=True):
            total_edge_counter += len(attribute)
            for key in attribute.keys():
                edge_types.add(key)
        print("number of edges: ", total_edge_counter)
        print("number of relations: ", len(edge_types))

        train_graph_sampler = GraphSampler(train_graph)

        def sample_train_graph_with_pattern(pattern):
            while True:

                sampled_train_query = train_graph_sampler.sample_with_pattern(pattern)

                train_query_train_answers = train_graph_sampler.query_search_answer(sampled_train_query)
                if len(train_query_train_answers) > 0:
                    break
            return sampled_train_query, train_query_train_answers


        print("sample training queries")



        for query_type, sample_pattern in original_query_types.items():


            print("test query_type: ", query_type)
            this_type_train_queries = {}

            n_query = n_queries_test_dict[data_dir]
            n_query = n_query // num_processes
            for _ in tqdm(range(n_query)):
                sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern(sample_pattern)

                this_type_train_queries[sampled_train_query] = {
                    "train_answers": train_query_train_answers,
                }


            with open("/home/data/jbai/sampled_data_301_train/" + data_dir + "_" + query_type + "_train_queries_" + str(id) + ".json", "w") as file_handle:
                json.dump(this_type_train_queries, file_handle)

if __name__ == "__main__":

    import time
    start_time = time.time()

    # Run the rest of the types
    with Pool(num_processes) as p:
        print(p.map(sample_301_test_data, range(num_processes)))

    print("--- %s seconds ---" % (time.time() - start_time))

