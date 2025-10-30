import bisect
from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        freq = Counter(power)
        sorted_values = sorted(freq.keys())
        n = len(sorted_values)
        if n == 0:
            return 0
        totals = [v * freq[v] for v in sorted_values]
        dp = [0] * n
        prefix_max = [0] * (n + 1)
        
        for i in range(n):
            current = sorted_values[i]
            lower = current - 3
            # Find the largest index j where sorted_values[j] <= lower using bisect_right on the subarray [0, i)
            j = bisect.bisect_right(sorted_values, lower, 0, i) - 1
            sum_take = totals[i]
            if j >= 0:
                sum_take += prefix_max[j + 1]
            sum_not_take = dp[i - 1] if i > 0 else 0
            dp[i] = max(sum_take, sum_not_take)
            prefix_max[i + 1] = max(prefix_max[i], dp[i])
        
        return dp[-1]