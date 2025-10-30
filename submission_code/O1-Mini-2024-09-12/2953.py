from collections import Counter
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = Counter(tuple(coord) for coord in coordinates)
        result = 0
        for (x, y), c in count.items():
            for a in range(k + 1):
                b = k - a
                x2 = x ^ a
                y2 = y ^ b
                if (x2, y2) in count:
                    if (x, y) < (x2, y2):
                        result += c * count[(x2, y2)]
                    elif (x, y) == (x2, y2):
                        result += c * (c - 1) // 2
        return result