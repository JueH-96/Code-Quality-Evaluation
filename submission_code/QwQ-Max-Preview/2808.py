class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        max_sum = 500 * 501  # Maximum possible sum of (time[i] + 1) for all walls
        dp = [float('inf')] * (max_sum + 1)
        dp[0] = 0  # Base case: 0 cost to achieve sum 0
        
        for i in range(n):
            current_time = time[i] + 1
            current_cost = cost[i]
            # Iterate backwards to avoid reusing the same wall multiple times
            for s in range(max_sum, -1, -1):
                if dp[s] != float('inf'):
                    new_s = s + current_time
                    if new_s > max_sum:
                        new_s = max_sum
                    if dp[new_s] > dp[s] + current_cost:
                        dp[new_s] = dp[s] + current_cost
        
        # The answer is the minimum cost for sums >= n
        return min(dp[n:max_sum + 1])