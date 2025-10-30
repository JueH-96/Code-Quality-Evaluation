from typing import List
import bisect
from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        if not count:
            return 0
        s = sorted(count.keys())
        n = len(s)
        val = [s[i] * count[s[i]] for i in range(n)]
        dp = [0] * n
        dp[0] = val[0]
        
        for i in range(1, n):
            option1 = dp[i-1]
            target = s[i] - 3
            idx = bisect.bisect_right(s, target) - 1
            if idx >= 0:
                current = val[i] + dp[idx]
            else:
                current = val[i]
            option2 = current
            dp[i] = max(option1, option2)
        
        return dp[-1]