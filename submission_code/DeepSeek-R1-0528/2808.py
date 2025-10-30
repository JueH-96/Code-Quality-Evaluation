class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            t_val = time[i] + 1
            current_cost = cost[i]
            for j in range(n, -1, -1):
                new_j = min(j + t_val, n)
                if dp[j] + current_cost < dp[new_j]:
                    dp[new_j] = dp[j] + current_cost
        return dp[n]