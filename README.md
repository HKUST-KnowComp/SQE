# Sequential Query Encoding (SQE)

The official implementation for paper Sequential Query Encoding For Complex Query Answering on Knowledge Graphs [[Paper]](https://arxiv.org/pdf/2302.13114.pdf).

The KG data we are using is from the KG reasoning repo from [here](http://snap.stanford.edu/betae/KG_data.zip). The data descriptions are here: https://github.com/snap-stanford/KGReasoning. Please put the downloaded files under <code>./KG_data</code> directory.

The complex query dataset for our benchmark can be downloaded [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/tzhengad_connect_ust_hk/EXgjlrPJHadPhPDQCuVFy88B-BCkdNJc1Mu1rTBURpfysQ?e=wCEFuo)（52.9GB).


We provided a wide range of baselines with our codebase. For experiments, please check out <code>example.sh</code> for script format. Also, you are welcomed to build your own models with our benchmark, by overriding the functions in <code>./models/model.py</code>.

During the running process, you can monitor the training process via tensorboard with following commands: <br>
<code> tensorboard --logdir your_log_dir --port the_port_you_fancy </code> <br>
<code> ssh -N -f -L localhost:port_number:localhost:port_number your_server_location </code>

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
