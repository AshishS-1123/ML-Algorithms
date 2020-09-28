import unittest
from ML_Algo.Preprocessing import StandardScaler
import numpy as np

# List containing the input data to the model
data = np.asarray( [0.96878163, 0.15967837, 0.49141737, 0.00389698, 0.88201058,\
                                    0.82652326, 0.5578806,  0.46005309, 0.37107311, 0.99547713] )
# List containing the expected data from the model
expected = np.asarray( [ 1.225483, -1.2714605, -0.24769312, -1.7522117, 0.957702,\
                                    0.7864646, -0.04258351, -0.34448528, -0.6190831, 1.3078669 ] )

class TestStandardScaler(unittest.TestCase):
    
    def test_Output(self):
        '''
        Test to check whether the StandardScaler gives correct value after transforming
        '''
        
        # Fit the Algo on the data
        StandardScaler.fit(data)
        # Get the transformed data
        actual = StandardScaler.transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Transformed data is not equal to expected values")
        
    def test_Mean(self):
        '''
        Test to check whether the mean of data calculated by the algorithm is correct
        '''
        
        # Fit the model on the data
        StandardScaler.fit(data)
        # Get the mean calculated by the model
        actual = StandardScaler.get_mean()
        # Get the actual mean using numpy
        expected_ = np.mean(data)
        # Check if both these value are almost equal
        areEqual = np.isclose(actual, expected_)
        # If they are not equal, print the required message
        self.assertTrue(areEqual, "Mean value found in model does no match the actual mean")
     
    def test_StdDev(self):
        '''
        Test to check whether the standard deviation found while fitting the data is correct
        '''
        
        # Fit the model on the data
        StandardScaler.fit(data)
        # Get the standard deviation calculated by the model
        actual = StandardScaler.get_std()
        # Get the actual standard deviation using numpy
        expected_ = np.std(data)
        # Check if both these value are almost equal
        areEqual = np.isclose(actual, expected_)
        # If they are not equal, print the required message
        self.assertTrue(areEqual, "Standard deviation found in model does no match the actual standard deviation")
       
    def test_FitAndTransform(self):
        '''
        Test to check whether applying the fit and transform method seperately gives correct result
        '''
        
        # Fit the Algo on the data
        StandardScaler.fit(data)
        # Get the transformed data
        actual = StandardScaler.transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Applying Fit and Transform methods seperately did not give correct results")
    
    def test_FitTransform(self):
        '''
        Tests to check whether applying the fit_transform method gives correct result
        '''
        
        # Fit and transform the data
        actual = StandardScaler.fit_transform(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Applying fit_transform method did not give correct results")
    
    def test_DataTypes(self):
        '''
        Test to check whether the methods work for different data types
        '''
        
        # Fit the model on int32 type data
        actual = StandardScaler.fit_transform(data.astype(np.int32))
        # Check if the model gave correct results
        self.assertTrue( actual.size != 0, "Functions did not work on 32 bit integer data type")
        # Fit the model on int8 type data
        actual = StandardScaler.fit_transform(data.astype(np.int8))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on 8 bit integer data type")
        # Fit the model on float type data
        actual = StandardScaler.fit_transform(data.astype(np.float))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on float data type")
        # Fit the model on double type data
        actual = StandardScaler.fit_transform(data.astype(np.double))
        # Check if the model gave the correct results
        self.assertTrue( actual.size != 0, "Functions did not work on double data type")
   
if __name__ == '__main__':
    unittest.main()