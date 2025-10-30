class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        max_effort = n * 501  # Maximum possible effective walls painted
        dp = [float('inf')] * (max_effort + 1)
        dp[0] = 0
        for i in range(n):
            c, t = cost[i], time[i]
            for j in range(max_effort - (t + 1), -1, -1):
                if dp[j] != float('inf'):
                    dp[j + t + 1] = min(dp[j + t + 1], dp[j] + c)
        result = min(dp[k] for k in range(n, max_effort + 1))
        return result