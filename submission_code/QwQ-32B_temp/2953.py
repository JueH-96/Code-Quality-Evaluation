from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        seen = defaultdict(int)
        count = 0
        for x, y in coordinates:
            for a in range(0, k + 1):
                b = k - a
                req_x = x ^ a
                req_y = y ^ b
                count += seen[(req_x, req_y)]
            seen[(x, y)] += 1
        return count