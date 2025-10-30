from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        seen = defaultdict(int)
        
        for x, y in coordinates:
            for target in range(k + 1):
                x_target = x ^ target
                y_target = y ^ (k - target)
                count += seen[(x_target, y_target)]
            seen[(x, y)] += 1
        
        return count