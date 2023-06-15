from ast import literal_eval as make_tuple
from sampling.sample import *
from tqdm import tqdm


def convert_s2t(text):
    # convert queries str into feedable str for make_tuple (i.e. no letter chars)
    dic = {
        'e': '0',
        'p': '1',
        'i': '2',
        'n': '3',
        'u': '4'
    }
    result = ""
    for i in text:
        if i in dic.keys():
            result += dic[i]
        else:
            result += i
    return result


########################################
# following functions convert negation queries to queries without negation
# 2in -> 1p
# 3in -> 2i
# inp -> 2p
# pin -> 2p
# pni -> 1p
def c_2in_1p(text):
    temp = make_tuple(convert_s2t(text))
    new_tuple = "(p,(" + str(temp[2][1]) + "),(e,(" + str(temp[2][2][1]) + ")))"
    return new_tuple


def c_3in_2i(text):
    temp = make_tuple(convert_s2t(text))
    new_tuple = "(i,(p,(" + str(temp[2][1]) + "),(e,(" + str(temp[2][2][1]) + "))),(p,(" + str(
        temp[3][1]) + "),(e,(" + str(temp[3][2][1]) + "))))"
    return new_tuple


def c_inp_2p(text):
    temp = make_tuple(convert_s2t(text))
    new_tuple = "(p,(" + str(temp[1]) + "),(p,(" + str(temp[2][2][1]) + "),(e,(" + str(temp[2][2][2][1]) + "))))"
    return new_tuple


def c_pin_2p(text):
    temp = make_tuple(convert_s2t(text))
    new_tuple = "(p,(" + str(temp[1][1]) + "),(p,(" + str(temp[1][2][1]) + "),(e,(" + str(temp[1][2][2][1]) + "))))"
    return new_tuple


def c_pni_1p(text):
    temp = make_tuple(convert_s2t(text))
    new_tuple = "(p,(" + str(temp[2][1]) + "),(e,(" + str(temp[2][2][1]) + ")))"
    return new_tuple


def check_negation(queries, convert_func, graph_sampler):
    identical = 0
    ta_origin = 0
    ta_convert = 0
    for i in tqdm(range(len(queries))):
        a = set()
        b = set()
        origin_answers = graph_sampler.query_search_answer(queries[i][0])
        a.update(origin_answers)
        ta_origin += len(a)

        new_answers = graph_sampler.query_search_answer(convert_func(queries[i][0]))
        b.update(new_answers)
        ta_convert += len(b)
        if (len(a) == len(b)):
            identical += 1
    print("Valid Negation Queries Rate: {}%, ({}/{})".format((len(queries) - identical) * 100 / len(queries),
                                                             (len(queries) - identical), len(queries)))
    print("Average answers of original queries: {}".format(ta_origin / len(queries)))
    print("Average answers of converted queries: {}".format(ta_convert / len(queries)))
    print("===============================================================")


if __name__ == '__main__':
    with open('./KG_data/FB15k-237-betae/sampled_query.pkl', 'rb') as f:
        data = pickle.load(f)
    # sampled_query format: list of sets of (query-answer) pairs
    dataset_path = "./KG_data/FB15k-237-betae/train.txt"
    train_graph = GraphConstructor(dataset_path)
    graph_sampler = GraphSampler(train_graph)
    # the following index is from 14-type sampled pkl file
    indices = [9, 10, 11, 12, 13]
    queries_2in = list(data[indices[0]])
    queries_3in = list(data[indices[1]])
    queries_inp = list(data[indices[2]])
    queries_pin = list(data[indices[3]])
    queries_pni = list(data[indices[4]])
    print("checking negation for 2in queries.")
    check_negation(queries_2in, c_2in_1p, graph_sampler)

    print("checking negation for 3in queries.")
    check_negation(queries_3in, c_3in_2i, graph_sampler)

    print("checking negation for inp queries.")
    check_negation(queries_inp, c_inp_2p, graph_sampler)

    print("checking negation for pin queries.")
    check_negation(queries_pin, c_pin_2p, graph_sampler)

    print("checking negation for pni queries.")
    check_negation(queries_pni, c_pni_1p, graph_sampler)
