from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        orig_zero = original[0]
        max_L = bounds[0][0]  # L_0 for i=0
        min_U = bounds[0][1]  # U_0 for i=0
        for i in range(1, len(original)):
            L_i = bounds[i][0] - original[i] + orig_zero
            U_i = bounds[i][1] - original[i] + orig_zero
            max_L = max(max_L, L_i)
            min_U = min(min_U, U_i)
        if max_L <= min_U:
            return min_U - max_L + 1
        else:
            return 0