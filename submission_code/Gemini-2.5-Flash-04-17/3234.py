from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        """
        Finds the indices i such that ((a_i^bi % 10)^ci) % m_i == target.

        Args:
            variables: A 0-indexed 2D array where variables[i] = [a_i, b_i, c_i, m_i].
            target: The target value to match the final result against.

        Returns:
            A list of indices i that satisfy the condition.
        """
        good_indices = [] # Initialize an empty list to store indices that meet the condition
        
        # Iterate through the input list 'variables'.
        # enumerate provides both the index 'i' and the list 'var_list' at that index.
        for i, var_list in enumerate(variables):
            # Unpack the four integers from the current sublist.
            a, b, c, m = var_list
            
            # Calculate the first part of the formula: (a^b) % 10.
            # Use Python's built-in pow(base, exp, mod) for efficient modular exponentiation.
            # This computes (a_i ** b_i) % 10.
            # Note: pow(a, b, 10) is equivalent to pow(a % 10, b, 10), but the built-in handles large 'a' correctly.
            intermediate_base = pow(a, b, 10)
            
            # Calculate the second part of the formula: (intermediate_base^c) % m.
            # This step calculates (intermediate_base ** c_i) % m_i.
            result = pow(intermediate_base, c, m)
            
            # Check if the final calculated result equals the target value.
            if result == target:
                # If the condition holds true, add the current index 'i' to the list of good indices.
                good_indices.append(i)
                
        # Return the list of all indices that satisfied the given condition.
        return good_indices