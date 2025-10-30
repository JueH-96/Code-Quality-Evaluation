class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            for j in range(n, 0, -1):
                if dp[j - 1] != float('inf'):
                    dp[j] = min(dp[j], dp[j - 1] + cost[i])
                if j > time[i]:
                    dp[j] = min(dp[j], dp[j - time[i] - 1])
        return min(dp)