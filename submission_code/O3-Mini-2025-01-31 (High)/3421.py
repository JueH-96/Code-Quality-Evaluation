from typing import List
from collections import defaultdict

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        remainder_counts = defaultdict(int)
        # Count the frequency of each remainder when divided by 24
        for hour in hours:
            remainder = hour % 24
            remainder_counts[remainder] += 1

        count = 0
        # Pairs with remainder 0 form complete days (0 + 0 = 0 mod 24)
        if remainder_counts[0] > 1:
            count += remainder_counts[0] * (remainder_counts[0] - 1) // 2
        
        # Pairs with remainder 12 form complete days (12 + 12 = 24 mod 24)
        if remainder_counts[12] > 1:
            count += remainder_counts[12] * (remainder_counts[12] - 1) // 2

        # For other remainders, pair remainder r with 24 - r, for r from 1 to 11 
        for r in range(1, 12):
            complement = 24 - r
            count += remainder_counts[r] * remainder_counts[complement]

        return count