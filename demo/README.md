# CLIP<sup>n</sup> Integration Notebook

This Jupyter Notebook demonstrates how to apply CLIP<sup>n</sup>to integrate heterogeneous high-content screening (HCS) datasets. The notebook guides users through a complete workflow that includes:

## 1. Data Loading and Preprocessing
-  Import multiple HCS datasets along with their metadata.
-  Preprocess high-dimensional phenotypic profiles to prepare them for integration.

## 2. CLIP<sup>n</sup> Model Application
-  Transform dataset-specific profiles into a shared latent space using individual encoders for each dataset.
-  Leverage overlapping reference compound categories and perform cross-dataset contrastive learning, rather than aligning datasets pairwise.

## 3. Visualization and Evaluation
-  Use dimensionality reduction techniques (e.g., UMAP) to visualize the integrated latent space.
- Assess the quality of integration with quantitative metrics such as total variation distance and F1 scores to ensure that samples from the same drug categories are properly aligned across datasets.

## 4. Transitive Predictions
- Map uncharacterized compounds into the unified latent space.
- Make transitive predictions of compound function by comparing their position to that of known reference compounds.

Overall, this notebook provides a comprehensive and reproducible pipeline for integrating and analyzing diverse HCS datasets, showcasing how CLIP<sup>n</sup> can unify disparate data sources to accelerate early drug discovery efforts.

## Contact

For questions or feedback, please contact steven.altschuler@ucsf.edu.

## License

This project is licensed under the MIT License.