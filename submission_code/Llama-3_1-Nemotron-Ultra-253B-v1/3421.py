from typing import List
from collections import defaultdict

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        count = defaultdict(int)
        total = 0
        for h in hours:
            rem = h % 24
            needed_rem = (24 - rem) % 24
            total += count[needed_rem]
            count[rem] += 1
        return total