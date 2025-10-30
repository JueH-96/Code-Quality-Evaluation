from typing import List
import bisect
from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count total damage for each unique spell value
        cnt = Counter(power)
        items = sorted((v, v * c) for v, c in cnt.items())
        
        # Separate values and their aggregated weights
        values = [v for v, w in items]
        weights = [w for v, w in items]
        n = len(values)
        
        # dp[i] = max damage using items[0..i]
        dp = [0] * n
        
        for i in range(n):
            # Find the rightmost index j < i with values[j] <= values[i] - 3
            # Because any value within ±2 is forbidden, so we need a gap ≥3
            target = values[i] - 3
            # bisect_right returns first index where values[idx] > target
            j = bisect.bisect_right(values, target) - 1
            
            # If there is a valid non-conflicting j, add dp[j], else just weights[i]
            take = weights[i] + (dp[j] if j >= 0 else 0)
            
            # Either skip this spell-value or take it
            dp[i] = take if i == 0 else max(dp[i-1], take)
        
        return dp[-1]