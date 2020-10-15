import unittest
import numpy as np

from ML_Algo.Preprocessing.Normalizer import NormL1, NormL2

# List containing the input data to the model
data = np.asarray( [ [1,2,3,4], [1,3,5,7] ] )
# List containing the expected data for l1 normalized data
expected_l1 = np.asarray( [ [0.1, 0.2, 0.3, 0.4], [0.0625, 0.1875, 0.3125, 0.4375] ] )
# List containing the expected data for l2 normalized data
expected_l2 = np.asarray( [ [0.18257418, 0.36514837, 0.5477225, 0.73029673], [0.10910894, 0.32732683, 0.5455447, 0.7637626] ] )

class TestNormalizer(unittest.TestCase):

    def test_Output_NormL1(self):
        '''
        Test to check whether the Normalizer gives correct value after transforming
        '''

        # Transform the data
        actual = NormL1(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected_l1)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Transformed data from NormL1 not equal to expected values")

    def test_Output_NormL2(self):
        '''
        Test to check whether the Normalizer gives correct value after transforming
        '''

        actual = NormL2(data)
        # Check if the calculated and expected values are equal (with a tolerance)
        areEqual = np.allclose(actual, expected_l2)
        # If the arrays are not equal, print the required message
        self.assertTrue( areEqual, "Transformed data from NormL2 not equal to expected values")



if __name__ == '__main__':
    unittest.main()
