class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        target_set = set(targetIndices)
        m = len(pattern)
        n = len(source)
        INF = float('inf')
        dp = [INF] * (m + 1)
        dp[0] = 0
        
        for i in range(n):
            for j in range(m, 0, -1):
                if source[i] == pattern[j - 1]:
                    cost = 1 if i in target_set else 0
                    if dp[j - 1] + cost < dp[j]:
                        dp[j] = dp[j - 1] + cost
        
        return len(targetIndices) - dp[m]