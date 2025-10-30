from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, var in enumerate(variables):
            a, b, c, m = var
            x = pow(a, b, 10)
            y = pow(x, c, m)
            if y == target:
                result.append(i)
        return result