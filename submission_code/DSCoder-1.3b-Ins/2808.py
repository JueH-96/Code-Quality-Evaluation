class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = max(dp[i + 1], cost[i] + (dp[i + time[i] + 1] if i + time[i] + 1 < n else float('inf')))
        return dp[0]