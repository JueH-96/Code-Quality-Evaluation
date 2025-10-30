from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Compute frequency of remainders when divided by 24
        freq = [0] * 24
        for h in hours:
            r = h % 24
            freq[r] += 1
        
        counter = 0
        for r in range(24):
            s = (24 - r) % 24
            if r < s:
                counter += freq[r] * freq[s]
            elif r == s:
                counter += freq[r] * (freq[r] - 1) // 2
        return counter