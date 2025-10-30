from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Count occurrences of each remainder modulo 24
        cnt = [0] * 24
        for h in hours:
            cnt[h % 24] += 1

        res = 0
        # Pairs where both hours % 24 == 0
        res += cnt[0] * (cnt[0] - 1) // 2
        # Pairs where remainders sum to 24 (r and 24-r)
        for r in range(1, 12):
            res += cnt[r] * cnt[24 - r]
        # Pairs where both hours % 24 == 12 (12+12 = 24)
        res += cnt[12] * (cnt[12] - 1) // 2

        return res