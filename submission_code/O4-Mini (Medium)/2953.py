from typing import List
from collections import Counter

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # We'll encode each (x,y) as a single integer key = (x << 20) | y,
        # since x,y <= 10^6 < 2^20. This makes hashmap lookups faster.
        SHIFT = 20
        # Build initial frequency map of all points
        freq = Counter(((x << SHIFT) | y) for x, y in coordinates)
        
        ans = 0
        for x, y in coordinates:
            key = (x << SHIFT) | y
            # remove current point from future counts
            freq[key] -= 1
            
            # try all splits a + b = k
            # we want (x ^ x2) = a and (y ^ y2) = b
            for a in range(k + 1):
                b = k - a
                x2 = x ^ a
                y2 = y ^ b
                desired_key = (x2 << SHIFT) | y2
                ans += freq.get(desired_key, 0)
        
        return ans