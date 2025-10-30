from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        low = []
        high = []
        for j in range(n):
            u, v = bounds[j]
            val = original[j]
            low_j = u - val
            high_j = v - val
            low.append(low_j)
            high.append(high_j)
        
        L = max(low)
        R = min(high)
        
        if L > R:
            return 0
        else:
            return R - L + 1