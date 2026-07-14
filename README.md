# CML (Cross-Modality Learning)
The official source code for paper **Cross-Modality Learning on Heavily Noisy
Attributed Graphs**,including methods T-CML (Topology-driven Cross-Modality Learning) and A-CML (Attribute-driven Cross-Modality Learning ).

## Environments
The proposed CML is implemented with python 3.8.8 on CPU.
All results in the paper are from running on an i7-10700 CPU.

No GPU required!
Use of the GPU may cause slowdowns.

### Packages
+ numpy==1.22.4
+ scipy==1.6.2
+ sklearn==0.24.1
+ matplotlib==3.3.4

```requirements.txt``` contains versions of all packages in our environment. 
You can install the same environment using the following command:
```pip install -r requirements.txt```

If you are using Anaconda, an identical environment can also be created by using the following command:
```conda env create -f environment.yml```


## Datasets
The 6 datasets we used: Cora, Citeseer, ACM, WiKi, DBLP, PubMed.
The ```data/``` holds several small datasets that can be used as demos. 
The full dataset can be accessed at https://drive.google.com/drive/folders/10Y2uqmQy21HPfgKBvxMov1svskxkOxXf?usp=sharing .
If you want to run the full dataset, just download all the data and put them in the ```data/``` directory.

All original attributed graph datasets is ```cora.mat```, ```citeseer.mat```, ```acm.mat```, ```dblp.mat```, ```pubmed.mat```. 

```cora_sorted.npz``` and ```citeseer_sorted.npz``` is the version after sorting by sample category, conveniently used to visualize matrix block diagonal effects.

```...Z1.npz``` is the result Z of completing the first stage of RAG for each dataset.
```...Z2.npz``` is the result Z of completing the second stage of RAG for each dataset.
These can used as the inputs in the multi-stage RAG ```RAGsSC.py```.

## Model
Our T-CML and A-CML model is in ```model.py```.


## Quick Start
Running node clustering of T-CML: ```python run_TCML.py```. Just select the dataset you need to run.

Running node clustering of multi-stage RAG: ```python TCMLs.py```. You need to select the dataset and stage number you want.

(Before running ```python TCMLs.py```, please make sure that the full dataset has been downloaded in Google Drive and saved in the ```data/``` directory.)


## Simulation Experiments
```subspace simulation/``` contains all subspace attributed graph simulation experiments.
```generate_data_X=XZ.py``` is to generate different noise simulation attributed graphs.
```xzdata.npz``` is the simulated attributed graph data we used in paper.

Running CML on simulated attributed graph: ```python RAG.py```.

Running SLSA on simulated attributed graph: ```python SLSA_A+X.py```.
