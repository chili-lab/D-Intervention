# [ICML 2025] Learning Distribution-wise Control in Representation Space for Language Models
This is the official implementation code for our paper [Learning Distribution-wise Control in Representation Space for Language Models].

Generally, we are inspired by the fact that intervention magnitude can be controlled by multiplying a concept vector with a scalar - so why don't we learn that distribution directly? In short, this is  a **deep latent variable model + intervention** research. 

We can directly learn a distribution in latent space for different tasks.

# Requirement

Our codebase is built on pyreft, please install the pyreft from pip:
```bash
pip install pyreft
```
And [`pyvene`](https://github.com/stanfordnlp/pyvene) is the backbone of pyreft library, where serve as great foundation to do intervention research.
# Intervention Type
We provide a set of intervention that could be choosed:
| Intervention Method | Description |
|-|-|
| [`RedIntervention`]() | Hadamard-Product Intervention  |
| `VIBRedIntervention` | Variational Information Bottleneck reduced intervention |
| [`LoReFT`](https://arxiv.org/abs/2404.03592) | Original Low-rank Representation Fine-Tuning intervention method |
| `DistributionalreftIntervention` | Distributional representation fine-tuning intervention |
| `VIBRawreftIntervention` | Variational Information Bottleneck raw representation intervention |
| `VIBAffinereftIntervention` | Variational Information Bottleneck affine representation intervention |
| `MiniTransformerIntervention` | Mini-transformer based intervention method |

# Training Scripts

Generally, if you want to train a distribution-wise intervention on math tasks, run:
```bash
python train.py -task math \
-data_dir dataset \
-model yahma/llama-7b-hf \
-seed 42 \
-l 0 -r 8 -p f7+l7 -e 9 -lr 3e-3 \
-type DistributionalreftIntervention \
-gradient_accumulation_steps 2 \
-batch_size 16 \
-eval_batch_size 4 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding
```

You can change `DistributionalreftIntervention` to any type above.

## Citation 
```bibtex
@misc{deng2025learningdistributionwisecontrolrepresentation,
      title={Learning Distribution-Wise Control in Representation Space for Language Models}, 
      author={Chunyuan Deng and Ruidi Chang and Hanjie Chen},
      year={2025},
      eprint={2506.06686},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.06686}, 
}
