from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        cnt = defaultdict(int)
        res = 0
        for x, y in coordinates:
            for a in range(k + 1):
                b = k - a
                target_x = x ^ a
                target_y = y ^ b
                res += cnt.get((target_x, target_y), 0)
            cnt[(x, y)] += 1
        return res