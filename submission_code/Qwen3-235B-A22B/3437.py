from bisect import bisect_right
from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        S = sorted(count.keys())
        n = len(S)
        if n == 0:
            return 0
        sum_s = [s * count[s] for s in S]
        dp = [0] * n
        dp[0] = sum_s[0]
        
        for i in range(1, n):
            target = S[i] - 3
            # Find the rightmost index where target is >= S[j]
            j = bisect_right(S, target) - 1
            current = sum_s[i]
            if j >= 0:
                current += dp[j]
            dp[i] = max(dp[i-1], current)
        
        return dp[-1]