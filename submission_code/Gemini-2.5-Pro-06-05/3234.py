from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        """
        Finds all indices `i` that satisfy the condition:
        ((a_i^b_i % 10)^c_i) % m_i == target
        """
        
        good_indices = []
        
        # Iterate through the variables array, getting both the index and the values.
        for i, (a, b, c, m) in enumerate(variables):
            
            # The formula involves two exponentiations. A naive calculation like a**b
            # could lead to extremely large numbers given the constraints, making it
            # inefficient. The correct and efficient approach is to use modular exponentiation.
            # Python's `pow(base, exp, mod)` function calculates (base**exp) % mod
            # without computing the full value of the power, which is ideal here.
            
            # Step 1: Calculate the inner part of the formula, (a^b % 10).
            term1 = pow(a, b, 10)
            
            # Step 2: Use the result from Step 1 to calculate the final value, (term1^c) % m.
            final_value = pow(term1, c, m)
            
            # Step 3: Check if the calculated value matches the target.
            if final_value == target:
                # If it matches, the index 'i' is considered "good".
                good_indices.append(i)
                
        return good_indices