# ReLiNet: Stable and Explainable Multistep Prediction with Recurrent Linear Parameter Varying Networks

This repository contains the necessary scripts to reproduce the results from our paper
"ReLiNet: Stable and Explainable Multistep Prediction with Recurrent Linear Parameter Varying Networks".

Clone this repository and install dependencies:
```shell
git clone https://github.com/AlexandraBaier/Supplement_ReLiNet.git
cd Supplement_ReLiNet
pip install .
```

All directories and files will be created within the cloned directory.

Run the following to download all datasets and set up the required directories:
```shell
python scripts/setup_environment.py
```

To run the experiments for the ship dataset run the following two scripts in order:
```shell
python scripts/run_experiment_ship_ind.py {device}
python scripts/run_experiment_ship_ood.py {device}
```
where `device` is the identifier (an integer starting at 0) for the GPU to run the experiments on. 
If you only have one GPU, set the value to `0`.

If these scripts are stopped for any reason, you can rerun them without issue. 
`run_experiment_ship_ind.py` remembers what models where already trained and validated.

To run the experiments for the Pelican dataset run the following script:
```shell
python scripts/run_experiment_industrial_robot.py {device}
```

Trained models are found in `models`, results in `results`, and datasets in `datasets`.
Environment variables pointing to the models, results, and configuration for each experiment are found in
`environment`.
