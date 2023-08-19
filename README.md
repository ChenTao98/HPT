## Hierarchical Prompt Tuning for Few-Shot Multi-Task Learning

This is code for Hierarchical Prompt Tuning for Few-Shot Multi-Task Learning

### Environment

python 3.7.11

torch 1.10.1

transformers 4.11.3

### Data

We use data in [GLUE](https://gluebenchmark.com/), and split raw data into different copies of few-shot data.

We provide a copy of 100-sample data in the directory of data

### Run

```
pip install -r requirements.txt
cd code
sh train.sh
```

