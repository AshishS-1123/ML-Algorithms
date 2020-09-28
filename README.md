<h1 style = "color: orange; text-align: right"> Machine Learning Algorithms From Scratch</h1>

<hr style = "background-color: lightblue; height: 2px">

### About this project

This project demonstrates how Machine Learning Algorithms can be implemented in C. These extensions have been built as C Extensions to Python, hence they have speed comparable to Python Packages like Scikit Learn.

### Directory Structure

- **ML_Algo** *( contains all the modules and code for the ML algorithms )*
    - **Preprocessing** *( constains extensions modules for data preprocessing )*
- **tests** *( contains scripts for testing the designed algorithms )*
    - **Preprocessing** *( contains unittests for preprocessing algorithms )*
- **setup.py** *( setup file for installing the package )*
- **README.md**

### Installation Instructions

1. Install NumPy as it is a dependency
2. Clone this repo and cd to it
3. Run ```python setup.py build``` to build the module
4. Run ```python setup.py install``` to install it

### Algorithms Implemented

- Preprocessing
    - MinMaxScaler
    - StandardScaler

