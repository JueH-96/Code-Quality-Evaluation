class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for c, t in zip(cost, time):
            v = t + 1
            for j in range(n, 0, -1):
                if j <= v:
                    dp[j] = min(dp[j], c)
                else:
                    dp[j] = min(dp[j], dp[j - v] + c)
        
        return dp[n]