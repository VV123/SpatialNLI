SpatialNLI | [Spatial Comprehension](./README.spatial.md) | [Annotation](./README.anno.md)

# SpatialNLI

  This repository is the full implementation of our paper [SpatialNLI: A Spatial Domain Natural Language Interface to Databases Using Spatial Comprehension](https://arxiv.org/pdf/1908.10917.pdf)
  
  This repository uses a Tensorflow implementation of this [paper](https://arxiv.org/abs/1809.02649). If you find this code useful in your research, please cite
```
@article{wang2018transfer,
  title={A transfer-learnable natural language interface for databases},
  author={Wang, Wenlu and Tian, Yingtao and Xiong, Hongyu and Wang, Haixun and Ku, Wei-Shinn},
  journal={arXiv preprint arXiv:1809.02649},
  year={2018}
}
```

## Dependencies
  - TF 1.4
  - python 2.7

## Usage

GeoQuery https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f

RestQuery http://www.cs.utexas.edu/users/ml/nldata/restquery.html 

Build data

```
USAGE
  $ python utils/data_manager.py

OPTIONS
  --data   Choose from ['geo'. 'rest']
```

Train 

```
USAGE
  $ python main.py

OPTIONS
  --mode    Choose from [train, infer] 
  --data    Choose from ['geo', 'rest']
```

## Evaluation

We use denotation match accuracy Acc<sub>dm</sub> for evaluation.

Geoquery

| Method        | Acc<sub>dm</sub>|
| ------------- | ------------- |
| ASN           | 87.1%         |
| SEQ2TREE      | 87.1%         |
| TRANX         | 88.2%         |
| JL16          | 89.2%         |
| **SpatialNLI**| [**90.4%**](https://drive.google.com/drive/folders/1GskZI_sPrDbp9yn6YjEtLmmEKtLvT85o)     |

Restaurant

|Method        | Acc<sub>dm</sub>|
|--------------|-----------------|
|**SpatialNLI**|[**100%**](https://drive.google.com/drive/folders/1heNxCCuQ2O8NgfIYFViG0lEk1KwF02Uq)        |


Geoquery + Rest

|Method| Acc<sub>dm</sub>|
|--------------|-----------------|
|**SpatialNLI**|[**90.7**](https://drive.google.com/drive/folders/1ydwkOq-2TokSgL3EmjmJC3i7oYL07PrO)|


  
