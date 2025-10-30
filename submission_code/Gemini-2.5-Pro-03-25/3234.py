import math
from typing import List

class Solution:
    """
    Solves the problem of finding good indices based on a modular exponentiation formula.
    """
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        """
        Finds indices i such that ((a_i^b_i % 10)^c_i) % m_i == target.

        The calculation involves two steps of modular exponentiation.
        Python's built-in pow(base, exponent, modulus) function is used 
        for efficient calculation, as it computes (base^exponent) % modulus 
        without calculating the full value of base^exponent, thus avoiding
        potential overflow issues with large numbers.

        Args:
            variables: A list of lists, where each inner list represents
                       [a_i, b_i, c_i, m_i] for the i-th index.
            target: The target integer value for the comparison.

        Returns:
            A list containing the indices i for which the formula holds true.
            The order of indices in the returned list does not matter.
        """
        
        good_indices = [] # Initialize an empty list to store the good indices

        # Iterate through the variables list using enumerate to get both the index (i) 
        # and the value (the list [a, b, c, m])
        for i, var_list in enumerate(variables):
            # Unpack the four integer values from the current inner list
            # Note: the problem uses a_i, b_i, c_i, m_i, but we use a, b, c, m 
            # within the loop for simplicity.
            a = var_list[0]
            b = var_list[1]
            c = var_list[2]
            m = var_list[3]

            # Step 1: Calculate (a^b % 10)
            # Use the built-in pow(base, exponent, modulus) for efficiency.
            # This calculates (a**b) % 10 without computing the full a**b.
            term1 = pow(a, b, 10)

            # Step 2: Calculate (term1^c % m)
            # Again, use pow() for efficient modular exponentiation.
            # This calculates (term1**c) % m.
            final_result = pow(term1, c, m)

            # Step 3: Check if the final result equals the target value
            if final_result == target:
                # If the condition holds, the current index i is "good"
                good_indices.append(i)

        # Return the list containing all the good indices found
        return good_indices