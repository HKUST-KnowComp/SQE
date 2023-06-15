from tqdm import tqdm

from collections import Counter

from sampling.sample import GraphConstructor

if __name__ == '__main__':
    with open("query_counting.txt", 'w') as fout:
        for dataset_path in ["./KG_data/FB15k-237-betae/train.txt",
                             "./KG_data/FB15k-betae/train.txt",
                             "./KG_data/NELL-betae/train.txt"]:
            print(dataset_path)
            fout.write(dataset_path + "\n")
            train_graph = GraphConstructor(dataset_path)

            # unique_relations_tails_map = Counter()
            # for head_node in tqdm(train_graph.nodes):
            #     tail_relations = set()
            #     for tail_node, attribute_dict in train_graph[head_node].items():
            #         for key in attribute_dict.keys():
            #             for tail_node_2, attribute_dict_2 in train_graph[head_node].items():
            #                 for key_2 in attribute_dict_2.keys():
            #                     if tail_node_2 == tail_node:
            #                         continue
            #                     tail_relations.add((key, tail_node, key_2, tail_node_2))
            #
            #     unique_relations_tails_map[head_node] = tail_relations
            #
            # three_intersection_count = 0
            # for head_node in tqdm(train_graph.nodes):
            #     tail_patterns = set()
            #     for tail_node, attribute_dict in train_graph[head_node].items():
            #         next_hop_patterns = unique_relation_tail_map[tail_node]
            #         for key in attribute_dict.keys():
            #             for pattern in next_hop_patterns:
            #                 if pattern[1] != head_node and pattern[3] != head_node:
            #                     tail_patterns.add((key, pattern[0], pattern[1], pattern[2], pattern[3]))
            #
            #     three_intersection_count += len(tail_patterns)
            #
            # print("3i total queries: ", three_intersection_count / 6)
            # fout.write("3i total queries: ")
            # fout.write(str(three_intersection_count / 6))
            # fout.write("\n")



            # for intersections
            unique_relation_tail_map = Counter()
            for head_node in tqdm(train_graph.nodes):
                tail_relations = set()
                for tail_node, attribute_dict in train_graph[head_node].items():
                    for key in attribute_dict.keys():
                        tail_relations.add((key,tail_node))

                unique_relation_tail_map[head_node] = tail_relations

            two_intersection_count = 0
            for head_node in tqdm(train_graph.nodes):
                tail_patterns = set()
                for tail_node, attribute_dict in train_graph[head_node].items():
                    next_hop_patterns = unique_relation_tail_map[tail_node]
                    for key in attribute_dict.keys():
                        for pattern in next_hop_patterns:
                            if pattern[1] != head_node:
                                tail_patterns.add((key, pattern[0], pattern[1]))

                two_intersection_count += len(tail_patterns)

            print("2i total queries: ", two_intersection_count /2 )
            fout.write("2i total queries: ")
            fout.write(str(two_intersection_count / 2))
            fout.write("\n")








            # for projections:
            one_projection_type_map = Counter()
            for head_node in tqdm(train_graph.nodes):
                tail_relations = set()
                for tail_node, attribute_dict in train_graph[head_node].items():
                    for key in attribute_dict.keys():
                        tail_relations.add(key)

                one_projection_type_map[head_node] = tail_relations

            print("1p total queries: ", sum([len(rel_set) for node, rel_set in one_projection_type_map.items()]))
            fout.write("1p total queries: ")
            fout.write(str(sum([len(rel_set) for node, rel_set in one_projection_type_map.items()])))
            fout.write("\n")

            two_projection_type_map = Counter()
            for head_node in tqdm(train_graph.nodes):
                relation_combinations = set()
                for tail_node, attribute_dict in train_graph[head_node].items():
                    for first_relation in attribute_dict.keys():
                        for second_relation in one_projection_type_map[tail_node]:
                            relation_combinations.add((first_relation, second_relation))

                two_projection_type_map[head_node] = relation_combinations

            print("2p total queries: ", sum([len(rel_set) for node, rel_set in two_projection_type_map.items()]))
            fout.write("2p total queries: ")
            fout.write(str(sum([len(rel_set) for node, rel_set in two_projection_type_map.items()])))
            fout.write("\n")

            three_projection_type_map = Counter()
            for head_node in tqdm(train_graph.nodes):
                relation_combinations = set()
                for tail_node, attribute_dict in train_graph[head_node].items():
                    for first_relation in attribute_dict.keys():
                        for second_relation in two_projection_type_map[tail_node]:
                            relation_combinations.add((first_relation, second_relation[0], second_relation[1]))

                three_projection_type_map[head_node] = relation_combinations
            print("3p total queries: ", sum([len(rel_set) for node, rel_set in three_projection_type_map.items()]))
            fout.write("3p total queries: ")
            fout.write(str(sum([len(rel_set) for node, rel_set in three_projection_type_map.items()])))
            fout.write("\n")

