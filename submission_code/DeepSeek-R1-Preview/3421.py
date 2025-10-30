from typing import List
from collections import defaultdict

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        remainder_freq = defaultdict(int)
        count = 0
        for hour in hours:
            r = hour % 24
            complement = (24 - r) % 24
            count += remainder_freq.get(complement, 0)
            remainder_freq[r] += 1
        return count