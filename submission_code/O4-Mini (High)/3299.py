from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = Counter(nums)
        maxVal = max(nums)
        
        # Handle x = 1 separately: all powers of 1 are 1, pattern length must be odd
        ones = counts.get(1, 0)
        if ones > 0:
            # If there are c ones, the longest odd ≤ c is c if c odd else c-1
            ans = ones if ones % 2 == 1 else ones - 1
        else:
            ans = 0
        
        # For each other base x > 1, try to build the longest [x, x^2, x^4, ..., x^(2^k), ...] chain
        for x, cnt in counts.items():
            if x == 1:
                continue
            
            T = 0         # how many levels we have with count >= 2
            j_stop = -1   # the highest exponent‐level j we can reach with count >= 1
            v = x
            
            # Build v_j = x^(2^j) by repeated squaring, stop when v not in counts
            while v <= maxVal and counts.get(v, 0) > 0:
                if counts[v] >= 2:
                    T += 1
                j_stop += 1
                v = v * v
            
            # We can choose k up to min(T, j_stop):
            #   for j < k we need 2 of v_j, for j = k we need 1 of v_k
            k = min(T, j_stop)
            length = 2 * k + 1
            ans = max(ans, length)
        
        # At least we can always pick one element
        return max(ans, 1)