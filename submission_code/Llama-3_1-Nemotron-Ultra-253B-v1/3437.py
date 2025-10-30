import bisect
from itertools import groupby
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        unique_sums = []
        for key, group in groupby(power):
            count = sum(1 for _ in group)
            unique_sums.append((key, key * count))
        
        if not unique_sums:
            return 0
        
        vals = [x[0] for x in unique_sums]
        sum_vals = [x[1] for x in unique_sums]
        n = len(vals)
        dp = [0] * n
        dp[0] = sum_vals[0]
        
        for i in range(1, n):
            target = vals[i] - 3
            j = bisect.bisect_right(vals, target) - 1
            prev_sum = dp[j] if j >= 0 else 0
            current = sum_vals[i] + prev_sum
            dp[i] = max(dp[i-1], current)
        
        return dp[-1]