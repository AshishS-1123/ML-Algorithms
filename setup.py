from setuptools import setup, Extension
import numpy as np

pre_processing_extension = Extension(name = "ML_Algo.Preprocessing.MinMaxScaler", sources = ["ML_Algo/Preprocessing/MinMaxScaler.c"])

setup(

    name = "ML_Algo",
    version = "0.0.0",
    description = "Machine Learning Algorithms implemented in C++",
    author = "Ashish Shevale",
    author_email = "shevaleashish@gmail.com",
    packages = [ 'ML_Algo.Preprocessing' ],
    ext_modules = [pre_processing_extension],
    include_dirs = [np.get_include()]
    
)
