from collections import Counter
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = Counter(coordinates)
        sum_total = 0
        for (x, y), cnt in freq.items():
            for a in range(k + 1):
                b = k - a
                x_prime = x ^ a
                y_prime = y ^ b
                sum_total += cnt * freq.get((x_prime, y_prime), 0)
        sum_i_j_equals_i = sum(freq.values()) if k == 0 else 0
        return (sum_total - sum_i_j_equals_i) // 2