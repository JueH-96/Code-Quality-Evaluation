from collections import defaultdict
from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = defaultdict(int)
        total = 0
        for h in hours:
            rem = h % 24
            comp = (24 - rem) % 24
            total += freq[comp]
            freq[rem] += 1
        return total