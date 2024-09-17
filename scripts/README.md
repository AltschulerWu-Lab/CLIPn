# Original scripts for reproducing main results in the paper

This folder contains the original scripts used to obtain the results in the paper. The scripts are organized in the following subfolders:

**Figure_2_simulation**: Generation and integration of simulated datasets.

**HCS datasets**: 
1. Integration of 13 HCS datasets generated from diverse experiments.
2. Prediction for unknown compounds using integrated reference compounds.


*Note*: 
1. We wrapped CLIPn into a Python package after those experiments. Now, we can directly import CLIPn by
```python
from clipn import clipn
```
2. We packed the curated datasets into a pickle file. You can directly load the datasets by
```python
import pickle
with open('picklefile.pkl', 'rb') as f:
    data = pickle.load(f)

X=data["X"]
y=data["y"]
```
Information for each study:
- `HCS_datasets.pkl`: 13 HCS datasets.
- `Hypoxia.pkl`: 6 hypoxia stress phenotypic profiles. 
- `Expression.pkl`: 2 L1000 transcript profiles and 6 phenotypical profiles. 


3. The UMAP visualization of embeddings can be slightly changing between two runs.