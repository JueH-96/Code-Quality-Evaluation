class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            # Calculate (a^b % 10)^c % m
            power_mod = pow(a, b, 10)  # a^b % 10
            result = pow(power_mod, c, m)  # (result)^c % m
            
            if result == target:
                good_indices.append(i)
        
        return good_indices