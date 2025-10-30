class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            
            # Calculate (a^b % 10)
            first = pow(a, b, 10)
            
            # Calculate ((a^b % 10)^c % m)
            final = pow(first, c, m)
            
            if final == target:
                result.append(i)
                
        return result