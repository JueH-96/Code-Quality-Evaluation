from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        for i, (a, b, c, m) in enumerate(variables):
            # Step-1: (a^b) % 10
            first_mod = pow(a, b, 10)
            
            # Step-2: ((a^b % 10)^c) % m
            final_mod = pow(first_mod, c, m)
            
            if final_mod == target:
                good_indices.append(i)
        
        return good_indices