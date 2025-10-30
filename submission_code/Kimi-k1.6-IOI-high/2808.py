class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            t = time[i] + 1
            c = cost[i]
            for s in range(n, -1, -1):
                if dp[s] != float('inf'):
                    new_s = min(s + t, n)
                    dp[new_s] = min(dp[new_s], dp[s] + c)
        
        return dp[n]