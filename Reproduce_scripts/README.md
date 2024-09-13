# Original scripts for reproducing main results in the paper

This folder contains the original scripts used to obtain the results in the paper. The scripts are organized in the following subfolders:

**Figure_2_simulation**: Generation and integration of simulated datasets.

**HCS datasets**: 
1. Integration of 13 HCS datasets generated from diverse experiments.
2. Prediction for unknown compounds using integrated reference compounds.


*Note*: 
1. We wrapped CLIPn into a Python package after those experiments. Now, we can directly import CLIPn by
```python
from clipn.model import clipn
```
2. We packed the curated datasets into a pickle file. You can directly load the datasets by
```python
import pickle
with open('datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)
```

3. The UMAP visualization of embeddings can slightly change.