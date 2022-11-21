# DialogueEIN

## Introduction

This is a reproduction on [DialogueEIN](https://aclanthology.org/2022.coling-1.57/): Emotion Interaction Network for Dialogue Affective Analysis-COLING22.

Project structure refers to @shenwzh3/DAG-ERC. Features and Dataset can be found there.

## Model

![image-20221111190813432](./README.assets/image-20221111190813432.png)

## Experiment Result

|                              | IEMOCAP | MELD  |
| ---------------------------- | ------- | ----- |
| **Ref**-weighted-avg-f-score | 68.93   | 65.37 |
| result-v1                    | 64.22   | 62.65 |
| result-SSC                   | 65.67   | 63.59 |
| result-v2                    | 62.83   | 63.3  |
| result-v3                    | 63.38   | 63.03 |
| result-v4                    | 63.84   | 63.1  |

## explanation

- V1: the inputs of the encoder in the semantic interaction network are featrues have been extracted by roberta-large in advance, fixed.
- SSC: remove DialogueEIN structure, substitute with a three layers MLP, hidden size the same with BERT config.hidden_size.
- V2: nearly the same as DialogueEIN raised in original paper, except separate learning rate and the linear transform layer after roberta.

- V3: add linear transform compared to v2
- V4: add separate learning rate compared to v3
