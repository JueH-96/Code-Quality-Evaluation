from collections import defaultdict

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = defaultdict(int)
        count = 0
        for hour in hours:
            r = hour % 24
            need = (24 - r) % 24
            count += freq[need]
            freq[r] += 1
        return count