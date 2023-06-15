import time

from sampling.sample import *

##################################
# this script takes sample.py functions to produce sampled pkl file.
# pkl format: list of set of q-a pairs
# [{set of (q-a) pairs for 1p} ,{{set of (q-a) pairs for 2p},..........]


if __name__ == '__main__':

    sample_amount = {
        "1p": 10,
        "2p": 10,
        "3p": 10,
        "2i": 10,
        "3i": 10,
        "ip": 10,
        "pi": 10,
        "2u": 10,
        "up": 10,
        "2in": 1000,
        "3in": 1000,
        "inp": 1000,
        "pin": 1000,
        "pni": 1000,
    }  # for testing

    query_types = {
        "1p": "(p,(e))",
        "2p": "(p,(p,(e)))",
        "3p": "(p,(p,(p,(e))))",  # added 3p
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

    dataset_path = "./KG_data/FB15k-237-betae/train.txt"
    output_path = "./KG_data/FB15K-237-betae/sampled_query.pkl"
    train_graph = GraphConstructor(dataset_path)
    print("sampling queries from: " + dataset_path)
    print("number of nodes: ", len(train_graph.nodes))
    print("number of head-tail pairs: ", len(train_graph.edges))

    total_edge_counter = 0
    edge_types = set()
    for u_node, v_node, attribute in train_graph.edges(data=True):
        total_edge_counter += len(attribute)
        for key in attribute.keys():
            edge_types.add(key)
    print("number of edges: ", total_edge_counter)
    print("number of relations: ", len(edge_types))

    graph_sampler = GraphSampler(train_graph)
    total_stime = time.time()
    set_list = []  # store 14 sets of q-a pairs
    for query_type, sample_pattern in query_types.items():
        stime = time.time()
        type_set = set()
        sep_time_array = [0.0, 0.0]
        print("processing query_type: ", query_type)

        while len(type_set) < sample_amount[query_type]:
            ts1 = time.time()
            initial_amount = len(type_set)
            query, answers = graph_sampler.sample_valid_question_with_answers(sample_pattern)
            ts2 = time.time()
            type_set.add((query, tuple(i for i in answers)))
            current_amount = len(type_set)
            ts3 = time.time()
            sep_time_array[0] += (float(ts2) - float(ts1))
            sep_time_array[1] += (float(ts3) - float(ts2))
        set_list.append(type_set)
        etime = time.time()
        time_len = float(etime) - float(stime)
        print("sampled " + str(sample_amount[query_type]) + " queries from type " + query_type + " in " + str(
            round(time_len, 3)) + " seconds. Set operation takes " + str(
            round((sep_time_array[1] / sum(sep_time_array) * 100), 2)) + "% time.")

    with open(output_path, 'wb') as f:
        pickle.dump(set_list, f)
    total_etime = time.time()
    total_time_len = float(total_etime) - float(total_stime)
    print("total_time_used: " + str(round(total_time_len, 3)) + " seconds.")
