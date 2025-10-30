from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = [0] * 24
        result = 0
        for h in hours:
            mod = h % 24
            complement = (24 - mod) % 24
            result += freq[complement]
            freq[mod] += 1
        return result