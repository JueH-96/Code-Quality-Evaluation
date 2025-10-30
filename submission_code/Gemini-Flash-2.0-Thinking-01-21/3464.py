class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        costs = {}
        for l in range(n):
            current_cost = 0
            for r in range(l, n):
                current_cost += nums[r] * ((-1) ** (r - l))
                costs[(l, r)] = current_cost
                
        dp = [0] * n
        for i in range(n):
            max_cost = -float('inf')
            for j in range(i + 1):
                current_cost = costs.get((j, i), 0)
                prefix_cost = 0
                if j > 0:
                    prefix_cost = dp[j - 1]
                total_cost = prefix_cost + current_cost
                max_cost = max(max_cost, total_cost)
            dp[i] = max_cost
            
        return dp[n - 1]