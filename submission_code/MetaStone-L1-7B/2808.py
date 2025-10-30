class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        max_sum = sum(time[i] + 1 for i in range(n))
        dp = [float('inf')] * (max_sum + 1)
        dp[0] = 0
        
        for i in range(n):
            weight = time[i] + 1
            cost_i = cost[i]
            for j in range(max_sum, weight - 1, -1):
                if dp[j - weight] + cost_i < dp[j]:
                    dp[j] = dp[j - weight] + cost_i
        
        min_cost = float('inf')
        for j in range(n, max_sum + 1):
            if dp[j] < min_cost:
                min_cost = dp[j]
        
        return min_cost if min_cost != float('inf') else 0