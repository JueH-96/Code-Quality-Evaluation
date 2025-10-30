from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        freq = [0] * 24
        for h in hours:
            r = h % 24
            freq[r] += 1
        
        total_pairs = 0
        total_pairs += freq[0] * (freq[0] - 1) // 2
        
        for r in range(1, 12):
            total_pairs += freq[r] * freq[24 - r]
        
        total_pairs += freq[12] * (freq[12] - 1) // 2
        
        return total_pairs