# Predictive Clinical Neuroscience Toolkit
Predictive Clinical Neuroscience software toolkit (formerly nispat). Methods for normative modelling, spatial statistics and pattern recognition. A quickstart guide for running some of the major functions is provided at the bottom of this page

## Basic installation (on a local machine)

i) install anaconda3 ii) create enviornment with "conda create --name env_name" iii) activate environment by "source activate env_name" iv) install required conda packages

```
conda install pip pandas scipy
```

v) install PCNtoolkit (plus dependencies)

```
pip install pcntoolkit
```

## Alternative installation (on a shared resource)
Make sure conda is available on the system.
Otherwise install it first from https://www.anaconda.com/ 

```
conda --version
```

Create a conda environment in a shared location

```
conda create -y python==3.7.7 numpy mkl blas --prefix=/shared/conda/normative_modeling/1.2.2
```

Activate the conda environment 

```
conda activate /shared/conda/normative_modeling/1.2.2
```

Install other dependencies

```
conda install -y pandas scipy 
```

Install pip dependencies

```
pip --no-cache-dir install nibabel sklearn torch glob3 
```

Clone the repo

```
git clone https://github.com/amarquand/PCNtoolkit.git
```

install in the conda environment

```
cd PCNtoolkit/
python3 setup.py install
```

Test 
```
python -c "import pcntoolkit as pk;print(pk.__file__)"
```

## Quickstart usage

For normative modelling, functionality is handled by the normative.py script, which can be run from the command line, e.g.

```
# python normative.py -c /path/to/training/covariates -t /path/to/test/covariates -r /path/to/test/response/variables /path/to/my/training/response/variables
```

For more information, please see the following resources:

* [documentation](https://github.com/amarquand/PCNtoolkit/wiki/PCNtookit-documentation)
* [developer documentation](https://amarquand.github.io/PCNtoolkit/doc/build/html/)
* a tutorial and worked through example on a [real-world dataset](https://github.com/predictive-clinical-neuroscience/PCNtoolkit-demo)
