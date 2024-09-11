# XY Model Lattice Generation using Metropolis Hastings Algorithm

This repository contains the code for generating lattices of an XY model using the Metropolis Hastings algorithm.

## Code Organization

* `get_data.py`: Main file for generating lattices of desired size.
* `xy.py`: Implements the Metropolis Hastings algorithm for XY model lattice generation.

## Usage

1. Run `get_data.py` to generate a lattice of desired size.
2. The lattice size can be adjusted by modifying the parameters in `get_data.py`.
3. The Metropolis Hastings algorithm is implemented in `xy.py` and is called by `get_data.py` to generate the lattice.

## Notes

* The XY model is a statistical mechanics model used to study phase transitions and critical phenomena.
* The Metropolis Hastings algorithm is a Markov chain Monte Carlo method used to sample from complex probability distributions.
* This code generates lattices of the XY model using the Metropolis Hastings algorithm.

## Dependencies

* Python 3.x
* NumPy
