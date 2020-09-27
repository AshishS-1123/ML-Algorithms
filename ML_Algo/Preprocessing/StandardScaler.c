#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

// Global Variables
int data_mean = 0;
int data_std = 0;

// Functions

/*
Function : fit
In Python : fit
Description : Go through the data and find the mean and standard deviation.
                        This will be the parameters our data is fit on.
Parameters : input_array - numpy array containing the data to be fit.
Return Value : None ( Python Object )
*/
PyObject* fit(PyObject* self, PyObject* args)
{
    PyArrayObject* input_array = NULL;
    // Parse the passed arguments and store the results in range_min and range_max variables
    if( ! PyArg_ParseTuple( args, "O", &input_array ) )
        return NULL;

    // Convert the data type of input_array so, the function becomes universal for all dtype arrays
    input_array = (PyArrayObject*) PyArray_Cast(input_array, NPY_FLOAT);
    
    // Store the dimensions of the input array
    int n_dims = PyArray_NDIM( input_array );
    
    // This functions will only work on single column arrays, warn the user about it
    if( n_dims != 1 )
        PyErr_SetString( PyExc_ValueError, "StandardScaler currently supports single column data only." );
    
    // Get the number of rows in the data
    int rows = (int) PyArray_SHAPE( input_array )[0];
    
    // Get the pointer to the data inside the numpy array
    float* data = (float*) PyArray_DATA( input_array );
    
    // Initialize the data_mean and data_std to zero.
    data_mean = 0, data_std = 0;
    
    // From the data, calculate the mean of all elements
    for(int i = 0; i < rows; ++i)
        data_mean += data[i];
        
    // data_mean currently holds the sum of elements of data,
    // So divide it by number of elements to get the mean
    data_mean /= rows;
    
    // Using the data elements and the mean, calculate the standard deviation
    for(int i = 0; i < rows; ++i)
        data_std += ( data[i] - data_mean ) * ( data[i] - data_mean );
        
    // data_std holds the sum of square of difference of elements and the mean
    // divide it by no of elements and take square root to get the standard devaition
    data_std = sqrt( data_std / rows );

    // Return the None Python Object
    return Py_None;
}

//########        MODULE LEVEL FUNCTIONS        ########

// method definitions
static PyMethodDef methods[] = {
  { "fit", fit, METH_VARARGS, "Fits the StandardScaler on the data given to it"},
  { "transform", transform, METH_VARARGS, "Transforms the data as per the StandardScaler formula"},
  {"fit_transform", fit_transform, METH_VARARGS, "Fits and Transforms the data at the same time"},
  { NULL, NULL, 0, NULL }
};

// module definition
static struct PyModuleDef StandardScaler_Mod = {
    PyModuleDef_HEAD_INIT,
    "StandardScaler",
    "Scales the data such that data has mean of 0 and standard deviation of 1.",
    -1,
    methods
};

// create the module
PyMODINIT_FUNC PyInit_StandardScaler(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&StandardScaler_Mod);
}

