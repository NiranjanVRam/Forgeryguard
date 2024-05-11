# Source code for: "ForgeryGuard: Web-based Image Authenticity Checker"


## Getting Started
Make sure your system has python 3.6.8rc1 version. If not, you can download it from [this](https://www.python.org/downloads/release/python-368rc1/) website.

Clone the github repo using:
git clone https://github.com/NiranjanVRam/Forgeryguard.git

Requirements have been mentioned at end of this file.

There are two ways of execution:
1. Through single module:
- run app.py directly from terminal or any IDE of your choice.
- Localhost opens automatically.
- Input image and click on process, wait for 3-8 minutes based on the input(not on size).

2. Through a series of program files:
Suggested IDE for execution: VS Code
- Run 0_compute_visualize_similarity_graph
- Run 1_forgery_detection
- Run 2_forgery_localization

While using this method, results(images, matrices, graphs, outputs) will directly be displayed in IDE itself.

## Evaluating the tool
Modules 3, 4 and 5 given below are for evaluation of the tool(ForgeryGuard).

It consumes more execution time and memory.

For 3, 4 and 5, you need to download each dataset below from its sources:
- Columbia
- Carvalho
- Korus

If any image from the dataset is not of .TIFF or .tiff format, change to it to same.


## Details for all the modules of the project
Please follow the examples in jupyter notebooks in the main directory for how to use this code.
- *0_compute_visualize_similarity_graph* shows how to compute the forensic similarity graphs
- *1_forgery_detection* shows how to compute forgery detection scores
- *2_forgery_localization* shows how to compute forgery localization prediction masks
- *3_compute_and_save_simgraph_for_benchmark_DBs* computes the forensic graphs for the three tampering datasets and saves to disk, to be used in subsequent notebooks
- *4_benchmark_forgery_detection* reproduces forgery detection results 
- *5_benchmark_forgery_localization* reproduces forgery localization results 

Note: the script *3a_format_dbs.sh* should be run prior to running notebooks 3-5. Alternatively you can modify the notebooks according to your own file structure. 

## Requirements
- Python 3.6.8
- tensorflow 1.14.0 (gpu version recommended)
- pillow
- seaborn
- tqdm
- python-igraph (available from conda-forge)
- python flask
- other requirements to be satisfied as per requirements.txt; depends on what all libraries have already been installed on your device.
