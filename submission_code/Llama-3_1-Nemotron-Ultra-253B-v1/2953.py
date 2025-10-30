from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = defaultdict(int)
        total = 0
        for a, b in coordinates:
            for s in range(k + 1):
                t = k - s
                x = a ^ s
                y = b ^ t
                total += freq[(x, y)]
            freq[(a, b)] += 1
        return total