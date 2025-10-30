from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        count = Counter(power)
        keys = sorted(count.keys())
        n = len(keys)
        
        if n == 0:
            return 0
        if n == 1:
            return keys[0] * count[keys[0]]
        
        dp = [0] * n
        dp[0] = keys[0] * count[keys[0]]
        
        for i in range(1, n):
            dp[i] = keys[i] * count[keys[i]]
            if keys[i] - keys[i - 1] > 2:
                dp[i] += dp[i - 1]
            else:
                if i >= 2 and keys[i] - keys[i - 2] > 2:
                    dp[i] += dp[i - 2]
                if i >= 3 and keys[i] - keys[i - 3] > 2:
                    dp[i] = max(dp[i], dp[i - 3] + keys[i] * count[keys[i]])
                if i >= 4:
                    dp[i] = max(dp[i], dp[i - 4] + keys[i] * count[keys[i]])
        
        return dp[-1]