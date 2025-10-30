from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        # Iterate through each list of variables [a_i, b_i, c_i, m_i]
        # The index of this list in 'variables' is 'i'
        for i in range(len(variables)):
            # Unpack the four components from variables[i]
            # variables[i] is of the form [a_i, b_i, c_i, m_i]
            a_i, b_i, c_i, m_i = variables[i]
            
            # The formula to check is: ((a_i^b_i % 10)^c_i) % m_i == target
            
            # Step 1: Calculate (a_i^b_i % 10)
            # The built-in pow(base, exp, mod) function computes (base^exp) % mod efficiently.
            # This is crucial for handling potentially large intermediate values of a_i^b_i.
            # The result, term1, will be an integer from 0 to 9.
            # Constraints state b_i >= 1.
            term1 = pow(a_i, b_i, 10)
            
            # Step 2: Calculate (term1^c_i % m_i)
            # Similarly, use pow() for ((a_i^b_i % 10)^c_i) % m_i.
            # term1 is the base, c_i is the exponent, m_i is the modulus.
            # Constraints state c_i >= 1.
            # The result, final_value, is the left-hand side of the formula.
            final_value = pow(term1, c_i, m_i)
            
            # Step 3: Compare with target
            # If the calculated final_value equals target, then index 'i' is good.
            if final_value == target:
                good_indices.append(i)
                
        return good_indices