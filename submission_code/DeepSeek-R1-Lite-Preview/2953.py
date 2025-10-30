from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = defaultdict(int)
        for x, y in coordinates:
            freq[(x, y)] += 1
        
        if k == 0:
            total = 0
            for count in freq.values():
                total += count * (count - 1) // 2
            return total
        else:
            total = 0
            for (x, y), count in freq.items():
                for a in range(0, k + 1):
                    b = k - a
                    x_prime = x ^ a
                    y_prime = y ^ b
                    total += freq.get((x_prime, y_prime), 0) * count
            return total // 2