from bisect import bisect_right
from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        sorted_values = sorted(count.keys())
        n = len(sorted_values)
        if n == 0:
            return 0
        sums = [val * count[val] for val in sorted_values]
        dp = [0] * n
        dp[0] = sums[0]
        for i in range(1, n):
            target = sorted_values[i] - 3
            pos = bisect_right(sorted_values, target)
            j = pos - 1
            sum_prev = dp[j] if j >= 0 else 0
            sum_take = sums[i] + sum_prev
            dp[i] = max(dp[i-1], sum_take)
        return dp[-1]