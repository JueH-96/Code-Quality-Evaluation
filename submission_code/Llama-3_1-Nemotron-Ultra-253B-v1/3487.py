from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        target_set = set(targetIndices)
        m = len(pattern)
        dp = [float('inf')] * (m + 1)
        dp[0] = 0
        for idx, c in enumerate(source):
            for i in range(m-1, -1, -1):
                if c == pattern[i]:
                    cost = 1 if idx in target_set else 0
                    dp[i+1] = min(dp[i+1], dp[i] + cost)
        return len(targetIndices) - dp[m]