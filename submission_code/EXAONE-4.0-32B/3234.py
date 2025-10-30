from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for idx, var in enumerate(variables):
            a, b, c, m = var
            base1 = pow(a, b, 10)
            result = pow(base1, c, m)
            if result == target:
                good_indices.append(idx)
        return good_indices