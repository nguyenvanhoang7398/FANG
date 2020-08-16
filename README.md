# Factual News Graph (FANG)
This is the implementation of FANG - a graph representation learning framework for fake news detection. For more details, please refer to our paper.
Van-Hoang Nguyen, Kazunari Sugiyama, Preslav Nakov, Min-Yen Kan, FANG: Leveraging Social Context for Fake News Detection Using Graph Representation (CIKM 2020)


## Installation
```bash
conda env create -f environment. yml 
```

## Requirements
### Packages
* conda 4.8.2
* python 3.7.7
* torch 1.5.1
* tensorboard 1.15.0
### Hardware
* GPU: Titan RTX 24220MiB total memory
* CPU: 16GiB total memory

## Data
We provided the processed data used in our experiments in `data/news_graph`

| Description | File name | Format |
| ----- | ----- | ----- |
| Social entities | entities.txt | |
| Social entity's features | entity_features.tsv | entity_id [tab] feature_1_value [tab] feature_2_value... |
| User-news support interactions with negative sentiment | support_negative.tsv | user_id [tab] news_id [tab] seconds_since_publication |
| User-news support interactions with neutral sentiment | support_neutral.tsv | user_id [tab] news_id [tab] seconds_since_publication |
| User-news deny interactions | deny.tsv | user_id [tab] news_id [tab] seconds_since_publication |
| User-news report interactions | report.tsv | user_id [tab] news_id [tab] seconds_since_publication |
| News information | news_info.tsv | news_id [tab] labels [tab] title |
| Indicator whether certain pair of entities should be closed or far, only used for evaluation, not for as labels | rep_entities.tsv | entity_1_id [tab] entity_2_id [tab] closed/far [tab] stance |
| Source-source citation interactions | source_citation.tsv | source_1_id [tab] source_2_id |
| Source-news publication interactions | source_publication.tsv | source_id [tab] news_id |
| User-user friendship interactions | user_relationships.tsv | user_1_id [tab] user_2_id |
| Train-val-test splits (representative of a fold) | train_test_{training percentage}.json | {"train": train_entities, "val": validate_entities, "test": test_entities} | 

Unprocessed data, including news and users who engage them can be found in `data/fang_fake.csv` and `data/fang_real.csv`.

## Run Graph Learning Frameworks
```
usage: run_graph.py [-h] [-t TASK] [-m MODEL] [-p PATH] [--percent PERCENT]
                    [--temporal] [--use-stance] [--use-proximity]
                    [--epochs EPOCHS] [--attention]
                    [--pretrained_dir PRETRAINED_DIR]
                    [--pretrained_step PRETRAINED_STEP]

Graph Learning

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  task name
  -m MODEL, --model MODEL
                        model name
  -p PATH, --path PATH  path to dataset
  --percent PERCENT
  --temporal            whether to use temporality
  --use-stance          whether to use stance
  --use-proximity       whether to use proximity
  --epochs EPOCHS       number of epochs
  --attention           whether to use attention
  --pretrained_dir PRETRAINED_DIR
                        path to pre-trained model directory
  --pretrained_step PRETRAINED_STEP
                        pre-trained model step
```

Training FANG for `30` epochs at `90%` data with `temporality`, `stance loss` and `proximity loss`.
```
python run_graph.py -t fang -m graph_sage -p data/news_graph --percent 90 --epochs=30 --attention --use-stance --use-proximity --temporal
```

Training GCN baseline for `1000` epochs at `90%` data.
```
python run_graph.py -t news_graph -m gcn -p data/news_graph --percent 90 --epochs=1000
```

## Other resources
* Relation filtering, Stance detection, Sentiment Classification models can be found [here](https://github.com/nguyenvanhoang7398/FANG-helper)
* Social media retriever used to crawl unprocessed data, implemented by Kai Shu et al. can be found [here](https://github.com/KaiDMML/FakeNewsNet/)
