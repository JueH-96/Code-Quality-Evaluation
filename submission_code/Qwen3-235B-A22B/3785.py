from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        O0 = original[0]
        max_a = -float('inf')
        min_b = float('inf')
        n = len(original)
        
        for i in range(n):
            u, v = bounds[i]
            o_i = original[i]
            a = u - o_i + O0
            b = v - o_i + O0
            if a > max_a:
                max_a = a
            if b < min_b:
                min_b = b
        
        if max_a > min_b:
            return 0
        return min_b - max_a + 1