from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = max(dp[i + 1], cost[original.index(source[i])] + dp[i + 1])
        return dp[0] if dp[0] == cost[original.index(target)] else -1