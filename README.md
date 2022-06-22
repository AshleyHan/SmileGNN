# SmileGNN
Han X, Xie R, Li X, et al. SmileGNN: Drug–Drug Interaction Prediction Based on the SMILES and Graph Neural Network[J]. Life, 2022, 12(2): 319.
![image](https://github.com/AshleyHan/SmileGNN/blob/3d6820b3cbf594dc2161699eac24cb706d3af1e7/SmileGNN.PNG)

## Requirement
See requirement.txt

## Installation and Usage
It is suggested to build a conda environment.\
Git clone the whole project.\
pip install -r requirement.txt  
python run.py

## Datasets
You can find KEGG data that has been processed as a knowledge graph [here](https://github.com/zhenglinyi/KGNN/tree/master/raw_data/kegg).\
The unprocessed PDD data can be found [here](https://github.com/xjtushilei/pdd_data_set).\
The SMILES are downloaded from DrugBank.

## Citation
Please cite the following paper if you find this repository useful.
```  
@article{han2022smilegnn,
  title={SmileGNN: Drug--Drug Interaction Prediction Based on the SMILES and Graph Neural Network},
  author={Han, Xueting and Xie, Ruixia and Li, Xutao and Li, Junyi},
  journal={Life},
  volume={12},
  number={2},
  pages={319},
  year={2022},
  publisher={MDPI}
}
```
