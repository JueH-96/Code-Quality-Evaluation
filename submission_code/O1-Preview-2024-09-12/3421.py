from typing import List
class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        modulo = 24
        counts = [0] * modulo
        for h in hours:
            rem = h % modulo
            counts[rem] += 1
        
        total_pairs = counts[0] * (counts[0] -1) // 2  # Pairs where remainder is 0
        total_pairs += counts[12] * (counts[12] - 1) // 2  # Pairs where remainder is 12 (since 24 is even)
        
        for r in range(1, 12):  # For remainders from 1 to 11
            total_pairs += counts[r] * counts[modulo - r]
        
        return total_pairs