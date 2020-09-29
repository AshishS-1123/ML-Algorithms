#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

//##############        FUNCTIONS        ##############

PyObject* norm_l2( PyObject* self, PyObject* args )
{
    // Create a NumPy artay object to hold the input array
    PyArrayObject* input_array = NULL;
    
    // Parse the input arguments and get the input array
    if( ! PyArg_ParseTuple(args, "O", &input_array ) )
        return NULL;
    
    // Cast the input_array to dtype float so the function can now be used for all data types
    input_array = (PyArrayObject*) PyArray_Cast(input_array, NPY_FLOAT);
    
    // Get the dimensions of the array object and store it
    npy_intp ndims = PyArray_NDIM( input_array );
    
    // Check whether the dimensions provided by the user are of correct dimensions
    switch( ndims )
    {
        // If the user provided a scaler or empty array, inform the user through error
        case 0: PyErr_SetString( PyExc_ValueError, "Cannot process empty data");
                      break;
        // if the user provided a 1d array, it does not need normalising as we need at lest 2 features
        case 1: PyErr_SetString( PyExc_ValueError, "Data with only 1 observation does no need to be normalized");
                      break;
        // If the user provided a 2d array, its ok
        case 2: break;
        // If the user provided array with any other dimension, inform user through error
        default: PyErr_SetString( PyExc_ValueError, "Data must be 2 dimensional only");
    }
    
    // Create a variable to hold the number of rows in the array
    npy_intp* shape = PyArray_SHAPE( input_array );
    
    // Create a pointer and point it to the data of the array
    float* data = (float*) PyArray_DATA( input_array );
    
    // Create a pointr to store the transformed data and assign space to it
    float* out = (float*) malloc( shape[0] * shape[1] * sizeof( float ) );
    
    // Repeat the following for every observation in the data
    for( int i = 0; i < shape[0]; ++i )
    {
        // STEP 1 : : CALCULATE THE  NORM i.e. sqrt( x1 ^ 2 + x2 ^ 2 ... xn ^ 2 )
        
        // Create a variable to hold the norm
        float norm = 0;
        
        // Loop through every element and add its square to norm
        for( int j = 0; j < shape[1]; ++j )
        {
            norm += data[ i * shape[1] + j ] * data[ i * shape[1] + j ];
        }
        
        // Take square root of norm to get the final norm
        norm = sqrt( norm );
        
        // STEP 2 : : TRANSFORM THE DATA ACCORDING TO THE NORM
        
        // Loop through every element of observation and transform it
        for(int j = 0; j < shape[1]; ++j)
        {
            out[ i * shape[1] + j ] = data[ i * shape[1] + j ] / norm;
        }
        
    }
    
    // Create a Python-NumPy array object to store the transformed output
        PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNewFromData(2, shape, NPY_FLOAT, (void*) out );
        
        return PyArray_Return( output );
        
}

//########        MODULE LEVEL FUNCTIONS        ########

// method definitions
static PyMethodDef methods[] = {
  {"NormL2", norm_l2, METH_VARARGS, "Normalizes observations with L2 Norm"},
  { NULL, NULL, 0, NULL }
};

// module definition
static struct PyModuleDef Normalizer_Mod = {
    PyModuleDef_HEAD_INIT,
    "Normalizer",
    "Rescales values accross indivisual observations to have unit norm",
    -1,
    methods
};

// create the module
PyMODINIT_FUNC PyInit_Normalizer(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&Normalizer_Mod);
}
