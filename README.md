# Kernel-Matrix Determinant Estimation from stopped Cholesky Decomposition
This is the code repository for the JMLR submission of the same name [pdf](./blob/master/Kernel-Matrix_Determinant_Estimates_from_stopped_Cholesky_Decomposition.pdf).
## Requirements
This software has been tested unter Ubuntu 18.04 using Python 3.8.

## Installation
Open a terminal in the projects home directory.
As a first step, we recommend to create a dedicated python environment. For example, using conda 
```
conda create --prefix ./env
conda activate ./env
```

Then all pacakges can be installed with the following command:
```
conda install -y -c conda-forge --file requirements.txt
```

After that, switch to the ``openblas`` folder and run the ``make`` command:
```
cd openblas
make
```
This may take a while. After that, run
```
make install PREFIX=../choleskies/openblas/
```
to complete the installation process.

### Troubleshooting
If the ``make`` command fails, try ``make FC=gfortran``.

## Running Experiments
Open a terminal in the projects home directory.
For the fast PUMADYN experiments, run 
```
python run_pumadyn.py
```
which executes on your local computer.

For the computationally more demanding experiments, our run scripts assume that you are in an environment which allows to send slurm [1] jobs to a cluster.
IMPORTANT: the order of the run scripts below is necessary and strictly sequential--only start the next script when the results of all previous results have been obtained.
```
python run_default_configurations.py
python run_blas_pivoted_configurations.py
python run_blas_stopped_configurations.py
```
If you want to reproduce the results for the GPyTorch stochastic trace estimator, first install GPyTorch:
```
conda install -y -c conda-forge --file optional_requirements.txt
```
then run:
```
python run_default_ste_configurations.py
```

## Result Generation
Open a terminal in the projects home directory. 
Run 
```
python make_figures.py
python make_bound_figures.py
```
from the current folder in a terminal.
This may take a while.
If you ran the GPyTorch experiments, you can generate the corresponding tikz pictures with
```
python make_ste_figures.py
```
To generate a ```figures.pdf```, switch to the ```tex``` folder and execute the following five commands (you can speed up ``make`` by adding the option ``-j number_of_cores_you_want_to_dedicate``):
```
cd tex
lualatex figures.tex
make -f figures.makefile
lualatex figures.tex
make -f figures.makefile -B
lualatex figures.tex
```

## Our Results
To use our results, open a terminal in the ``results`` folder and run 
```
cat diagonals_part* > diagonals.zip
```
Then extract ``diagonals.zip`` and ``mlruns.zip`` in the folder ``results``.

[1] https://slurm.schedmd.com
