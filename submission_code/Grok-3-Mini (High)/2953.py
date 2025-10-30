import collections
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = collections.Counter(map(tuple, coordinates))
        n = len(coordinates)
        S = 0
        for coord, cnt in freq.items():
            x, y = coord
            C = 0
            for dy in range(k + 1):
                dx = k - dy
                target_x = x ^ dx
                target_y = y ^ dy
                C += freq.get((target_x, target_y), 0)
            S += cnt * C
        if k > 0:
            return S // 2
        else:
            return (S - n) // 2