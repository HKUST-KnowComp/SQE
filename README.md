# SQE

The official implementation for paper Sequential Query Encoding For Complex Query Answering on Knowledge Graphs [[Paper]](https://arxiv.org/pdf/2302.13114.pdf).

The KG data we are using is from the KG reasoning repo from [here](http://snap.stanford.edu/betae/KG_data.zip). The data descriptions are here: https://github.com/snap-stanford/KGReasoning. Please put the downloaded files under <code>./KG_data</code> directory.

The complex query dataset for our benchmark can be downloaded [here]().


We provided a wide range of baselines with our codebase. For experiments, please check out <code>example.sh</code> for script format. Also, you are welcomed to build your own models with our benchmark, by overriding the functions in <code>./models/model.py</code>.

During the running process, you can monitor the training process via tensorboard with following commands: <br>
<code> tensorboard --logdir your_log_dir --port the_port_you_fancy </code> <br>
<code> ssh -N -f -L localhost:port_number:localhost:port_number your_server_location </code>

If you find the code/data/paper interesting, please cite our paper!

```
@misc{bai2023sequential,
      title={Sequential Query Encoding For Complex Query Answering on Knowledge Graphs}, 
      author={Jiaxin Bai and Tianshi Zheng and Yangqiu Song},
      year={2023},
      eprint={2302.13114},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
