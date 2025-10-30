class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            
            # Calculate (a^b % 10)
            first_part = pow(a, b, 10)
            
            # Calculate ((a^b % 10)^c) % m
            second_part = pow(first_part, c, m)
            
            # Check if it equals target
            if second_part == target:
                result.append(i)
        
        return result