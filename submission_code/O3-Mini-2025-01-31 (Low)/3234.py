from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute the last digit of (a^b)
            last_digit = pow(a, b, 10)
            # Raise last_digit to the power c modulo m
            result = pow(last_digit, c, m)
            if result == target:
                good_indices.append(i)
        return good_indices