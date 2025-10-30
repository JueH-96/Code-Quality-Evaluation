from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # We want pairs (i, j) such that (hours[i] + hours[j]) % 24 == 0.
        # We can reduce each hour modulo 24, then count each remainder.
        mods = [0] * 24
        for h in hours:
            mods[h % 24] += 1

        count = 0
        # Pairs with both numbers having remainder 0
        count += mods[0] * (mods[0] - 1) // 2
        
        # Since 24 is even, remainder 12 pairs among themselves yield (12 + 12) % 24 == 0.
        count += mods[12] * (mods[12] - 1) // 2
        
        # For other remainders, pair remainder r with remainder (24 - r)
        for r in range(1, 12):
            count += mods[r] * mods[24 - r]
        
        return count