# CLIP<sup>n</sup> Integration Notebook

This repository contains the Jupyter Notebook `CLIPn_integration.ipynb`, which demonstrates how to use CLIP<sup>n</sup>—a contrastive deep-learning framework—to integrate heterogeneous high-content screening (HCS) datasets.

## Notebook Workflow

- **Data Loading and Preprocessing:**  
  The notebook loads multiple HCS datasets along with their metadata and performs the preprocessing steps as implemented in the code to prepare the phenotypic profiles for integration.

- **CLIP<sup>n</sup> Model Application:**  
  The notebook applies the CLIP<sup>n</sup> model to transform dataset-specific profiles into a unified latent space. It leverages overlapping reference compound categories to perform cross-dataset contrastive learning.

- **Visualization and Evaluation:**  
  The integrated latent space is visualized using dimensionality reduction (e.g., UMAP). Evaluation metrics such as total variation distance and F1 scores are calculated to assess the quality of the integration.

- **Transitive Predictions:**  
  The notebook demonstrates how to map uncharacterized compounds into the unified latent space and predict their functions based on the proximity to known reference compounds.

## How to Run the Notebook

- **Open the Notebook:**  
  Launch Jupyter Notebook or JupyterLab in the directory containing `CLIPn_integration.ipynb` and open the file.

- **Execute Cells Sequentially:**  
  Run each cell by pressing **Shift+Enter** in the order presented. The notebook is organized into clearly labeled sections corresponding to the workflow.

- **Review Outputs:**  
  Check the latent representation `z` (e.g., visualizations and evaluation metrics) for downstream analysis.

## Contact

For questions or feedback, please contact [steven.altschuler@ucsf.edu](mailto:steven.altschuler@ucsf.edu).

## License

This project is licensed under the MIT License.