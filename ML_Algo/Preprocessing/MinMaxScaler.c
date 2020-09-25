#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<Python.h>
#include<numpy/arrayobject.h>

// Global Variables

// variable to hold the minimum value the transformed data should have
int range_min = 0;
// variable to hold the maximum value the transformed data should have
int range_max = 1;

// variable to hold the smallest element in the data
int min_x = 1e5;
// variable to hold the largest element in the data
int max_x = 0;

// Functions

/*
Function : min_max_scaler
In Python : set_range
Description : Set the variables for minimum values the transformed data is allowed to have
Parameters : range_min - the smallest value the transformed data is allowed to have
                        range_max - the largest value the transfprmed data is allowed to have
Return Value : None ( Python Object )
*/
PyObject* min_max_scaler(PyObject* self, PyObject* args)
{
    // Parse the passed arguments and store the results in range_min and range_max variables
    if( ! PyArg_ParseTuple( args, "ii", &range_min, &range_max ) )
        return NULL;

    // Return the None Python Object
    return Py_None;
}

/*
Function : fit
In Python : fit
Description : Go through the data and find the least and highest values in the data.
                        These will be the parameters for out model
Parameters : input_array - numpy array containing the data to be fit
Return Value : None ( Python Object )
*/
PyObject* fit(PyObject* self, PyObject* args )
{
    // Create a python array object to hold the data passed
    PyArrayObject* input_array = NULL;
    
    // Parse the parameters and store the data passed into the input_array variable
    if( !PyArg_ParseTuple( args, "O", &input_array ) )
        return NULL;
    
    // Convert the data type of input_array so, the function becomes universal for all dtype arrays
    input_array = (PyArrayObject*) PyArray_Cast(input_array, NPY_FLOAT);
    
    // Store the dimensions of the input array
    int n_dims = PyArray_NDIM( input_array );
    
    // This functions will only work on single column arrays, warn the user about it
    if( n_dims != 1 )
        PyErr_SetString( PyExc_ValueError, "MinMaxScaler currently supports single column data only." );
    
    // Get the number of rows in the data
    int rows = (int) PyArray_SHAPE( input_array )[0];
    
    // Get the pointer to the data inside the numpy array
    float* data = (float*) PyArray_DATA( input_array );
    
    // Loop through every element in data, and look for the minimum and maximum values in it
    for( int i = 0; i < rows; ++i )
    {
        // if the current element is less than the current minimum value in the row,
        if( data[ i ] < min_x )
            // that element will be our minimum
            min_x = data[ i ];
            
        // if the current element is larger than the current miximum in the row,
        if( data[ i ] > max_x )
            // that element will be out maximum
            max_x = data[ i ];
    }
    
    // Return the None Python Object
    return Py_None;
}

/*
Function : transform
In Python : transform
Description : Transform the data according to the MinMaxScaler formula
                         and the min and max values found in the fit 
Parameters : data - the data to be transformed
Return Value : Numpy array containing the transformed data
*/
PyObject* transform(PyObject* self, PyObject* args )
{
    // Create a Python Array Object to hold the data passed as arguments
    PyArrayObject* input_array = NULL;
    
    // Parse the arguments and store the data in input_array
    if( !PyArg_ParseTuple( args, "O", &input_array ) )
        return NULL;
    
    // Convert the dtype of the array to float, so our function works for all data type arrays
    input_array = (PyArrayObject*) PyArray_Cast(input_array, NPY_FLOAT);
    
    // Get the number of dimensions of the array
    int n_dims = PyArray_NDIM( input_array );
    
    // This function will only work on single column arrays, so warn the user
    if( n_dims != 1 )
        PyErr_SetString( PyExc_ValueError, "MinMaxScaler currently supports single column data only." );
        
    // Get the number of rows in the array
    int rows = (int) PyArray_SHAPE( input_array )[0];
    
    // Get the pointer to the data in the numpy array
    float* data = (float*) PyArray_DATA( input_array );
    
    // Create a pointer to store the result and assign memory to it
    float* out_data = (float*) malloc( rows * sizeof(float) );
    
    // Go through every element of data
    for( int i = 0; i < rows; ++i )
    {
        // Apply the MinMaxScaler formula to every element
        out_data[ i ] = ( ( range_max - range_min ) * ( data[ i ] - min_x ) ) / ( max_x - min_x ) + range_min;
    }
    
    // Create a new Python Array Object that stores the transformed data
    PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNewFromData(1, PyArray_DIMS( input_array ), NPY_FLOAT, (void*) out_data);
    
    // Return the Python-Numpy Array Object
    return PyArray_Return( output );
}

/*
Function : fit_transform
In Python : fit_transform

*/
PyObject* fit_transform(PyObject* self, PyObject* args )
{
    PyArrayObject* input_array = NULL;
    
    if( !PyArg_ParseTuple( args, "O", &input_array ) )
        return NULL;
    
    input_array = (PyArrayObject*) PyArray_Cast(input_array, NPY_FLOAT);
    
    int n_dims = PyArray_NDIM( input_array );
    
    if( n_dims != 1 )
        PyErr_SetString( PyExc_ValueError, "MinMaxScaler currently supports single column data only." );
        
    int rows = (int) PyArray_SHAPE( input_array )[0];
    
    float* data = (float*) PyArray_DATA( input_array );
    
    for( int i = 0; i < rows; ++i )
    {
        if( data[ i ] < min_x )
            min_x = data[ i ];
            
        if( data[ i ] > max_x )
            max_x = data[ i ];
    }
    
    float* out_data = (float*) malloc( rows * sizeof(float) );
    
    for( int i = 0; i < rows; ++i )
    {
        out_data[ i ] = ( ( range_max - range_min ) * ( data[ i ] - min_x ) ) / ( max_x - min_x ) + range_min;
    }
    
    PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNewFromData(1, PyArray_DIMS( input_array ), NPY_FLOAT, (void*) out_data);
    
    return PyArray_Return( output );
    
}

//########        MODULE LEVEL FUNCTIONS        ########

// method definitions
static PyMethodDef methods[] = {
  {"set_range", min_max_scaler, METH_VARARGS, "Sets the range between which the transformed values should appear"},
  { "fit", fit, METH_VARARGS, "Fits the MinMaxScaler on the data given to it"},
  { "transform", transform, METH_VARARGS, "Transforms the data as per the MinMaxScaler formula"},
  {"fit_transform", fit_transform, METH_VARARGS, "Fits and Transforms the data at the same time"},
  { NULL, NULL, 0, NULL }
};

// module definition
static struct PyModuleDef MinMaxScaler_Mod = {
    PyModuleDef_HEAD_INIT,
    "MinMaxScaler",
    "Scales the data such that all elements are in the given range",
    -1,
    methods
};

// create the module
PyMODINIT_FUNC PyInit_MinMaxScaler(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&MinMaxScaler_Mod);
}

