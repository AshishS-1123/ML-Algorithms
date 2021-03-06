from setuptools import setup, Extension
import numpy as np

####################### EXTENSIONS #######################

# MinMaxScaler Extension in Preprocessing
PrePro_MinMax_Extension = Extension(name = "ML_Algo.Preprocessing.MinMaxScaler", sources = ["ML_Algo/Preprocessing/MinMaxScaler.c"])

# Standard Scaler Extension in Preprocessing
PrePro_Std_Extension = Extension(name = "ML_Algo.Preprocessing.StandardScaler", sources = ["ML_Algo/Preprocessing/StandardScaler.c"])

# Normalizer Extension in Preprocessing
Norm_Extension = Extension(name = "ML_Algo.Preprocessing.Normalizer", sources = ["ML_Algo/Preprocessing/Normalizer.c"])

# Create a list containing all the extensions
ext_modules = [
                            PrePro_MinMax_Extension,
                            PrePro_Std_Extension,
                            Norm_Extension
]

####################### SETUP FOR PACKAGE #######################

setup(

    name = "ML_Algo",
    version = "0.0.0",
    description = "Machine Learning Algorithms implemented in C++",
    author = "Ashish Shevale",
    author_email = "shevaleashish@gmail.com",
    packages = [ 'ML_Algo.Preprocessing' ],
    ext_modules = ext_modules,
    include_dirs = [np.get_include()]
    
)
