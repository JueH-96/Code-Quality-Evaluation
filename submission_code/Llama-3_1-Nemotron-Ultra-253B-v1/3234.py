from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            x = pow(a, b, 10)
            y = pow(x, c, m)
            if y == target:
                result.append(i)
        return result