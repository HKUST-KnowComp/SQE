

import pandas as pd

all_query_types = pd.read_csv("../preprocess/test_generated_formula_anchor_node=3.csv").reset_index(drop = True)#debug

print(all_query_types)


all_query_types_filtered =  all_query_types.iloc[29:, :]
print(all_query_types_filtered)

sampled_types = all_query_types_filtered.sample(n = 29)
print(sampled_types)


concated_df = pd.concat([all_query_types.iloc[:29, :], sampled_types]).reset_index(drop = True)
print(concated_df)
concated_df.to_csv("../preprocess/test_generated_formula_anchor_node=3_filtered.csv")

