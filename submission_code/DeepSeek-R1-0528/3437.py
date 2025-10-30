import bisect
from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        cnt = Counter(power)
        unique_damages = sorted(cnt.keys())
        n = len(unique_damages)
        if n == 0:
            return 0
        
        dp = [0] * n
        
        for i in range(n):
            d = unique_damages[i]
            total = d * cnt[d]
            x = d - 3
            idx = bisect.bisect_right(unique_damages, x)
            prev_idx = idx - 1
            prev_dp = dp[prev_idx] if prev_idx >= 0 else 0
            skip_val = dp[i-1] if i > 0 else 0
            dp[i] = max(skip_val, total + prev_dp)
        
        return dp[-1]