from typing import List
from collections import Counter
import bisect

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        sorted_unique = sorted(count.keys())
        n = len(sorted_unique)
        if n == 0:
            return 0
        totals = [v * count[v] for v in sorted_unique]
        dp = [0] * n
        dp[0] = totals[0]
        
        for i in range(1, n):
            current_val = sorted_unique[i]
            current_total = totals[i]
            target = current_val - 3
            # Find the rightmost value in sorted_unique[0..i-1] that is <= target
            prev_idx = bisect.bisect_right(sorted_unique, target, 0, i) - 1
            take = (dp[prev_idx] if prev_idx >= 0 else 0) + current_total
            not_take = dp[i-1]
            dp[i] = max(take, not_take)
        
        return dp[-1]