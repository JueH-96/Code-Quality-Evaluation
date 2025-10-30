from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        # Iterate through the variables array with index
        for i, var_set in enumerate(variables):
            # Unpack the variables [a_i, b_i, c_i, m_i]
            a, b, c, m = var_set
            
            # Step 1: Calculate (a_i^b_i) % 10
            # Use Python's built-in pow(base, exp, mod) for efficient modular exponentiation.
            # This avoids calculating a_i^b_i directly, which could be a very large number.
            first_mod_result = pow(a, b, 10)
            
            # Step 2: Calculate (first_mod_result^c_i) % m_i
            final_result = pow(first_mod_result, c, m)
            
            # Step 3: Check if the final result matches the target
            if final_result == target:
                good_indices.append(i)
                
        return good_indices