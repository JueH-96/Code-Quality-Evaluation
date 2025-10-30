from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for idx, (a, b, c, m) in enumerate(variables):
            # Compute first value: (a^b) % 10 using built-in pow modulus to avoid large numbers.
            first = pow(a, b, 10)
            # Compute second value: (first^c) % m.
            result = pow(first, c, m)
            if result == target:
                good_indices.append(idx)
        return good_indices