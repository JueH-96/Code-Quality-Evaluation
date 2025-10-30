from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        """
        This function takes a 2D array variables and an integer target as input.
        It returns a list of indices where the given formula holds true.

        The formula is: ((a_i^b_i % 10)^c_i) % m_i == target
        """
        
        # Initialize an empty list to store the good indices
        good_indices = []
        
        # Iterate over the variables array with index and value
        for i, var in enumerate(variables):
            a, b, c, m = var  # unpack the values
            
            # Calculate the value of the formula
            formula_value = pow(pow(a, b, 10), c, m)
            
            # Check if the formula value equals the target
            if formula_value == target:
                # If it does, add the index to the good indices list
                good_indices.append(i)
        
        # Return the list of good indices
        return good_indices