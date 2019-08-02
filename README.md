SpatialNLI | [Spatial Comprehension](./README.spatial.md) | [Annotation](./README.anno.md)

# SpatialNLI

  SpatialNLI: A Spatial Domain Natural Language Interface to Databases Using Spatial Comprehension
  
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

To annotate Geo880

```python utils/annotate/annotate_geo.py```

To annotate Rest

```python utils/annotate/annotate_rest.py```

To Build data

```python utils/data_manager.py --data 'geo'```

```python utils/data_manager.py --data 'rest'```

To train new model

```python main.py --mode train --data 'geo'```

```python main.py --mode train --data 'rest'```

To infer using pre-trained model

```python main.py --mode infer --data 'geo'```

```python main.py --mode infer --data 'rest'```

To augment Geo880

```python utils/augmentation/augmentation_geo.py```

To augment Rest

```python utils/augmentation/augmentation_rest.py```

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


  
