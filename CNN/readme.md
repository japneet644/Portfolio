# Implementation of CNN Baseline for Physics Generation

This repository contains the implementation of the CNN baseline model described in the paper "[Towards meaningful physics from generative models](https://arxiv.org/abs/1705.09524)" by Cristoforetti, Marco, et al. This code was used as a baseline for our research paper.

## Code Organization

* `CNNbaseline.py`: Main code file implementing the CNN baseline model.
* `data_loader_special.py`: Loads the data for CNN training.
* `get_parameters.py`: Generates lattice data using Markov Chain Monte Carlo (MCMC). The lattice size can be adjusted.

## Usage

1. Adjust the lattice size in `get_parameters.py` as needed.
2. Run `get_parameters.py` to generate the lattice data.
3. Run `CNNbaseline.py` to train the CNN baseline model. The code will invoke `data_loader_special.py` to load the data for CNN training.

## Notes

* This implementation is based on the paper "[Towards meaningful physics from generative models](https://arxiv.org/abs/1705.09524)" by Cristoforetti, Marco, et al.
* This code was used as a baseline for our research paper.
* The lattice size can be adjusted in `get_parameters.py` to change the size of the generated data.

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* SciPy
