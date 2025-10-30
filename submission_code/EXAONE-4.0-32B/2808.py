class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            v = time[i] + 1
            c = cost[i]
            old_dp = dp.copy()
            for j in range(n + 1):
                if old_dp[j] == float('inf'):
                    continue
                new_j = j + v
                if new_j > n:
                    new_j = n
                if old_dp[j] + c < dp[new_j]:
                    dp[new_j] = old_dp[j] + c
        
        return dp[n]