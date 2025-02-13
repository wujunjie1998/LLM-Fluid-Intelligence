[![arXiv](https://img.shields.io/badge/arXiv-2502.07190-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2502.07190) [![Web](https://img.shields.io/badge/Web-ARAOC-blue.svg?style=plastic)](https://wujunjie1998.github.io/araoc-benchmark.github.io/)

This repository contains the code and data of the paper:

> Understanding LLMs’ Fluid Intelligence Deficiency: An Analysis of the ARC Task
> 
> [Junjie Wu](https://wujunjie1998.github.io/), [Mo Yu](https://sites.google.com/site/moyunlp/), [Lemao Liu](https://lemaoliu.github.io/), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung), [Jie Zhou](https://openreview.net/profile?id=~Jie_Zhou8)



## Data

The `data` folder consists of two main parts used in this paper. The first part, `data/ARC`, contains the 100 ARC tasks we employ, while the second, `data/ARAOC`, includes the ARAOC benchmark we developed in this work along with additional data for ARAOC-related experiments.

## Example Data

In this paper, we apply a matrix-format input to represent the 2D-grid inputs of ARC and ARAOC tasks for LLMs. Below is an example of an ARAOC task (Change Color), which includes three input-output pairs as in-context examples and a testing input-output pair to evaluate LLMs.


```
Example input grid:
[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5, 5], [0, 0, 0, 0, 0, 5, 5]]
Example output grid:
[[0, 0, 0, 0, 5, 5, 0], [0, 0, 0, 0, 5, 5, 0], [0, 0, 0, 0, 0, 0, 0]]

Example input grid:
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 7, 7], [0, 0, 7, 7], [0, 0, 7, 7], [0, 0, 0, 0]]
Example output grid:
[[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

Example input grid:
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
Example output grid:
[[0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

Testing input grid:
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
Testing output grid:
[[0, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```



## Run the Experiments

### Evaluation

#### GPT models
To evaluate GPT models on ARC tasks, run
```
python arc_gpt.py
```
You can also use natural language input by
```
python arc_gpt.py --language
```

To run GPT models on ARAOC tasks, run 
```
python arc_gpt.py --task_type
```
where `task_type` is one of `{move, copy, change_color, mirror, fill_internal, scale}`
To run GPT models on additional ARAOC-related experiments 
1. set `task_type` in one of `{move/copy_up*}` to run experiments in Table 6; 
2. set `task_type` in one of `{move_small/copy_small}` to run experiments in Table 7;
3. set `task_type` to `move+copy` to run experiment in Table 8;
4. set `task_type` in `{mirror_left, mirror_right}` to run experiments in Table 12.

You can also use natural language input by
```
python arc_gpt.py --task_type --language
```

### Evaluation

