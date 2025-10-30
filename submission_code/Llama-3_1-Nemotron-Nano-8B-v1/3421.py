from collections import defaultdict
from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = defaultdict(int)
        count = 0
        for h in hours:
            mod = h % 24
            complement = (24 - mod) % 24
            count += freq[complement]
            freq[mod] += 1
        return count