class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        for i, (a, b, c, m) in enumerate(variables):
            # Calculate (a^b % 10)
            ab_mod_10 = pow(a, b, 10)
            # Calculate (ab_mod_10^c % m)
            result = pow(ab_mod_10, c, m)
            
            if result == target:
                good_indices.append(i)
        
        return good_indices