from collections import defaultdict
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = defaultdict(int)
        ans = 0
        for a, b in coordinates:
            for t in range(k + 1):
                c = a ^ t
                d = b ^ (k - t)
                ans += freq.get((c, d), 0)
            freq[(a, b)] += 1
        return ans