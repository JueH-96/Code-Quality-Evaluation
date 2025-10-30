from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute a^b mod 10
            x = pow(a, b, 10)
            # Compute (x^c) mod m
            y = pow(x, c, m)
            if y == target:
                good_indices.append(i)
        return good_indices