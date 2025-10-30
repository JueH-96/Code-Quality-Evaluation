from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count_map = defaultdict(int)
        result = 0
        for x, y in coordinates:
            for a in range(k + 1):
                b = k - a
                x_prev = x ^ a
                y_prev = y ^ b
                result += count_map[(x_prev, y_prev)]
            count_map[(x, y)] += 1
        return result