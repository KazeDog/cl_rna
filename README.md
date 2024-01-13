# cl_rna
[![python](https://badge.ttsalpha.com/api?icon=python&label=python&status=3.8.13)](https://example.com)  [![pytorch](https://badge.ttsalpha.com/api?icon=pytorch&label=pytorch&status=1.8.1&color=yellow)](https://example.com)  [![cuda](https://badge.ttsalpha.com/api?icon=nvidia&label=cuda&status=11.3&color=green)](https://example.com)

## Project Overview

RNA modifications are important for deciphering the function of cells and their regulatory mechanisms. In recent years, researchers have developed many deep learning methods to identify specific modifications. However, these methods require model retraining for each new RNA modification and cannot progressively identify the newly identified RNA modifications. To address this challenge, we propose an innovative incremental learning framework that incorporates multiple incremental learning methods. Our experimental results confirm the efficacy of incremental learning strategies in addressing the RNA modification challenge. By uniquely targeting 10 RNA modification types in a class incremental setting, our framework exhibits superior performance. Notably, it can be extended to new category methylation predictions without the need for retraining with previous data, improving computational efficiency. Through the accumulation of knowledge, the model is able to evolve and continuously learn the differences across methylation, mitigating the problem of catastrophic forgetting during deep model training. Overall, our framework provides various alternatives to enhance the prediction of novel RNA modifications and illuminates the potential of incremental learning in tacking numerous genome data.


## Installation

### 1. Cloning the Project

First, you need to clone the project repository from GitHub to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/KazeDog/cl_rna.git
```

This command will create a copy of the   project in your current working directory.

### 2. Setting Up the Environment

After cloning the project, the next step is to set up the project environment. This project uses Conda, a popular package and environment management system. To create the environment with all the required dependencies, navigate to the project directory and run:

```bash
cd cl_rna
conda env create -f environment.yml
```

This command will read the environment.yml file and create a new Conda environment with the name specified in the file. It will also install all the dependencies listed in the file.

### 3. Activating the Environment

Once the environment is created, you need to activate it. To do so, use the following command:

```bash
conda activate cl_rna
```

Replace **cl_rna** with the actual name of the environment, as specified in the **environment.yml** file.


## Quick Start Guide

This guide provides instructions on how to run the deep learning program for training and testing purposes. The program allows for various operational modes and parameters to be set via command-line arguments.



```bash
   python train.py --si 
```


### Note

- You can replace `--si` with other methods such as `--lwf`.
- If no method is used, the baseline is executed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.