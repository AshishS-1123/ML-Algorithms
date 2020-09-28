import unittest
from ML_Algo.Preprocessing import MinMaxScaler
import numpy as np

# List containing the input data to the model
data = np.asarray( [0.96878163, 0.15967837, 0.49141737, 0.00389698, 0.88201058,\
                                    0.82652326, 0.5578806,  0.46005309, 0.37107311, 0.99547713] )
# List containing the expected data from the model
expected = np.asarray( [ 18.92311258, -13.71583253, -0.3335961, -20.0, 15.42279841,\
                                    13.18445906, 2.34750744, -1.59882028, -5.18824199, 20.0] )

class TestMinMaxScaler(unittest.TestCase):
    
    def test_Output(self):
        '''
        Test to check whether the MinMaxScaler gives correct value after transforming
        '''
        
        # Set the Minimum and Maximum range values for MinMaxScaler
        MinMaxScaler.set_range(-20, 20)
        # Fit the Algo on the data
        MinMaxScaler.fit(data)
        # Get the transformed data
        actual = MinMaxScaler.transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Transformed data not equal to expected values")
        
    def test_MinValue(self):
        '''
        Test to check whether the Minimum value found while fitting the data is correct
        '''
        
        # Fit the MinMaxScaler on the data, so the values for MIN are updated
        MinMaxScaler.fit(data)
        # Get the minimum value as found by MinMaxScaler
        actual = MinMaxScaler.get_min()
        # Get the sctual minimum value using numpy
        expected_ = np.amin(data)
        # Check if both these value are almost equal
        areEqual = np.isclose(actual, expected_)
        # If they are not equal, print the required message
        self.assertTrue(areEqual, "Minimum value found in model does no match the actual minimum value")
     
    def test_MaxValue(self):
        '''
        Test to check whether the Maximum value found while fitting the data is correct
        '''
        
        # Fit the MinMaxScaler on the data, so the value for MAX is updated
        MinMaxScaler.fit(data)
        # Get the maximum value as found by MinMaxScaler
        actual = MinMaxScaler.get_max()
        # Get the sctual minimum value using numpy
        expected_ = np.amax(data)
        # Check if both these value are almost equal
        areEqual = np.isclose(actual, expected_)
        # If they are not equal, print the required message
        self.assertTrue(areEqual, "Maximum value found in model does no match the actual maximum value")
       
    def test_FitAndTransform(self):
        '''
        Test to check whether applying the fit and transform method seperately gives correct result
        '''
        
        # Set the Minimum and Maximum range values for MinMaxScaler
        MinMaxScaler.set_range(-20, 20)
        # Fit the Algo on the data
        MinMaxScaler.fit(data)
        # Get the transformed data
        actual = MinMaxScaler.transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Applying Fit and Transform methods seperately did not give correct results")
    
    def test_FitTransform(self):
        '''
        Tests to check whether applying the fit_transform method gives correct result
        '''
        
        # Set the Minimum and Maximum range values for MinMaxScaler
        MinMaxScaler.set_range(-20, 20)
        # Fit and transform the data
        actual = MinMaxScaler.fit_transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Applying fit_transform method did not give correct results")
    
    def test_DataTypes(self):
        '''
        Test to check whether the methods work for different data types
        '''
        
        # Set the Minimum and Maximum range values for MinMaxScaler
        MinMaxScaler.set_range(-20, 20)
        # Fit the model on int32 type data
        actual = MinMaxScaler.fit_transform(data.astype(np.int32))
        # Check if the model gave correct results
        self.assertTrue( actual.size != 0, "Functions did not work on 32 bit integer data type")
        # Fit the model on int8 type data
        actual = MinMaxScaler.fit_transform(data.astype(np.int8))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on 8 bit integer data type")
        # Fit the model on float type data
        actual = MinMaxScaler.fit_transform(data.astype(np.float))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on float data type")
        # Fit the model on double type data
        actual = MinMaxScaler.fit_transform(data.astype(np.double))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on double data type")
   
if __name__ == '__main__':
    unittest.main()