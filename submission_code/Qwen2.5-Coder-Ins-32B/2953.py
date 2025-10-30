from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        seen = defaultdict(int)
        
        for x1, y1 in coordinates:
            for x in range(k + 1):
                x2 = x1 ^ x
                y2 = y1 ^ (k - x)
                count += seen[(x2, y2)]
            seen[(x1, y1)] += 1
        
        return count