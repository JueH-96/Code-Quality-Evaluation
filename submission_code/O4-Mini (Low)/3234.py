from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # First reduce a^b modulo 10
            x = pow(a, b, 10)
            # Then raise that result to the power c modulo m
            val = pow(x, c, m)
            if val == target:
                good_indices.append(i)
        return good_indices