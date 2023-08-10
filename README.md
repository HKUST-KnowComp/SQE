# Sequential Query Encoding (SQE)

The official implementation for the paper Sequential Query Encoding For Complex Query Answering on Knowledge Graphs [[Paper]](https://arxiv.org/pdf/2302.13114.pdf).

The KG data we are using is from the KG reasoning repo from [here](http://snap.stanford.edu/betae/KG_data.zip). The data descriptions are here: https://github.com/snap-stanford/KGReasoning. Please put the downloaded files under <code>./KG_data</code> directory.

The complex query dataset for our benchmark can be downloaded [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tzhengad_connect_ust_hk/EXgjlrPJHadPhPDQCuVFy88B-BCkdNJc1Mu1rTBURpfysQ?e=wCEFuo)（52.9GB).
Some people experience difficulty in downloading large files from onedrive on the command line. [Here](https://sushantag9.medium.com/download-data-from-onedrive-using-command-line-d27196a676d9) is a tutorial on downloading onedrive files in the command line. 


We provided a wide range of baselines with our codebase. For experiments, please check out <code>example.sh</code> for script format. 

During the running process, you can monitor the training process via tensorboard with following commands: <br>
<code> tensorboard --logdir your_log_dir --port the_port_you_fancy </code> <br>
<code> ssh -N -f -L localhost:port_number:localhost:port_number your_server_location </code>

## Supported Models:

Iterative Encoding Model:

| Model Flag (-m) | Paper  |
|---|---|
| gqe |  [Embedding logical queries on knowledge graphs](https://proceedings.neurips.cc/paper/2018/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html)  |
| q2b | [Query2box: Reasoning over knowledge graphs in vector space using box embeddings](https://openreview.net/forum?id=BJgr4kSFDS) |
| betae | [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://proceedings.neurips.cc/paper/2020/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html)  |
| hype | [Self-supervised hyperboloid representations from logical queries over knowledge graphs](https://dl.acm.org/doi/10.1145/3442381.3449974) |
| mlp / mlp_mixer| [Neural methods for logical reasoning over knowledge graphs](https://openreview.net/forum?id=tgcAoUVHRIB)  |
| cone | [Cone: Cone embeddings for multihop reasoning over knowledge graphs](https://openreview.net/pdf?id=Twf_XYunk5j) |
| q2p |  [Query2Particles: Knowledge Graph Reasoning with Particle Embeddings](https://aclanthology.org/2022.findings-naacl.207/) |
| fuzzqe | [Fuzzy Logic Based Logical Query Answering on Knowledge Graphs](https://arxiv.org/abs/2108.02390) |
| tree_lstm | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |
| tree_rnn | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |

Sequential Encoding Models:

| Model Flag (-m) | Paper  |
|---|---|
| biqe | [Answering Complex Queries in Knowledge Graphs with Bidirectional Sequence Encoders](https://arxiv.org/abs/2004.02596) |
| tcn | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |
| lstm | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |
| gru | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |
| transformer | (this paper) [Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |


## Brining your own Query Encoding Model!

Also, you are welcome to build your own models with our benchmark, by overriding the functions in <code>./models/model.py</code>. You only need to write your model, and the rest of things can be done by the code in this repo~

## Citations:
If you find the code/data/paper interesting, please cite our paper!

```
@article{
      bai2023sequential,
      title={Sequential Query Encoding for Complex Query Answering on Knowledge Graphs},
      author={Jiaxin Bai and Tianshi Zheng and Yangqiu Song},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2023},
      url={https://openreview.net/forum?id=ERqGqZzSu5},
      note={}
}
```
