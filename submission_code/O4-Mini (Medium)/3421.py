from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        """
        Count pairs (i, j) with i < j such that hours[i] + hours[j] is a multiple of 24.
        """
        n = len(hours)
        count = 0
        # We can work with remainders modulo 24 to simplify the check
        mods = [h % 24 for h in hours]
        for i in range(n):
            for j in range(i + 1, n):
                if (mods[i] + mods[j]) % 24 == 0:
                    count += 1
        return count