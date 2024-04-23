# Source code for: "Exposing Fake Images Forensic Similarity Graphs" 
by Owen Mayer and Matthew C. Stamm  
Deparment of Electrical and Computer Engineering  
Drexel University - Philadelphia, PA, USA

Visit the [project webpage](http://omayer.gitlab.io/forensicgraph/)

The [paper](https://ieeexplore.ieee.org/abstract/document/9113265) is available through the IEEE Journal of Selected Topics in Signal Processing 

## Getting Started

Please follow the examples in jupyter notebooks in the main directory for how to use this code.
- *0_compute_visualize_similarity_graph* shows how to compute the forensic similarity graphs and how to reproduce Figure 1 from the paper
- *1_forgery_detection* shows how to compute forgery detection scores
- *2_forgery_localization* shows how to compute forgery localization prediction masks
- *3_compute_and_save_simgraph_for_benchmark_DBs* computes the forensic graphs for the three tampering datasets and saves to disk, to be used in subsequent notebooks
- *4_benchmark_forgery_detection* reproduces forgery detection results 
- *5_benchmark_forgery_localization* reproduces forgery localization results 

Note: the script *3a_format_dbs.sh* should be run prior to running notebooks 3-5. Alternatively you can modify the notebooks according to your own file structure. 

## Requirements
- Python 3
- tensorflow 1.15 (gpu version recommended)
- pillow
- seaborn
- tqdm
- python-igraph (available from conda-forge)
- jupyter-notebook (for running examples)

## Notes
- The included models are trained on JPEG-aligned image patches. Analysis must be performed according to the 16x16 JPEG gridding of the image (if it is a JPEG). 
- The version of the paper currently (as of Nov 2020) on IEEE has captions on Table III and Table IV got swapped. It should be “Table III MCC, per image threshold” and “Table IV MCC, per database threshold.”


## Cite this code
If you are using this code for academic research, please cite this paper:

Mayer, Owen, and Matthew C. Stamm. "Exposing Fake Images with Forensic Similarity Graphs." *IEEE Journal of Selected Topics in Signal Processing* (2020).

bibtex:
```
@ARTICLE{mayer2020forensicgraphs,
  author={O. {Mayer} and M. C. {Stamm}},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Exposing Fake Images With Forensic Similarity Graphs}, 
  year={2020},
  volume={14},
  number={5},
  pages={1049-1064},}
```