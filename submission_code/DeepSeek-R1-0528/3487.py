class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        target_set = set(targetIndices)
        INF = 10**9
        dp = [INF] * (m + 1)
        dp[0] = 0
        
        for i in range(n):
            for j in range(m - 1, -1, -1):
                if dp[j] != INF and source[i] == pattern[j]:
                    cost = 1 if i in target_set else 0
                    if dp[j] + cost < dp[j + 1]:
                        dp[j + 1] = dp[j] + cost
        
        return len(targetIndices) - dp[m]