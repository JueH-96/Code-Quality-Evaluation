from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        freq_map = defaultdict(int)
        for i in range(len(coordinates)):
            x1, y1 = coordinates[i]
            for xor_x in range(k + 1):
                xor_y = k - xor_x
                x2 = x1 ^ xor_x
                y2 = y1 ^ xor_y
                count += freq_map[(x2, y2)]
            freq_map[(x1, y1)] += 1
        return count