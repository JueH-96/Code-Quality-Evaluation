from collections import Counter
import bisect
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        sorted_unique = sorted(count.keys())
        n = len(sorted_unique)
        if n == 0:
            return 0
        sums = [count[x] * x for x in sorted_unique]
        dp = [0] * n
        dp[0] = sums[0]
        for i in range(1, n):
            x = sorted_unique[i] - 3
            j = bisect.bisect_right(sorted_unique, x) - 1
            take = sums[i] + (dp[j] if j >= 0 else 0)
            not_take = dp[i-1]
            dp[i] = max(take, not_take)
        return dp[-1]