class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            max_cost_for_i = -float('inf')
            current_subarray_cost = 0
            for j in range(i, n):
                current_subarray_cost += nums[j] * (1 if (j - i) % 2 == 0 else -1)
                current_total_cost = current_subarray_cost + dp[j + 1]
                max_cost_for_i = max(max_cost_for_i, current_total_cost)
            dp[i] = max_cost_for_i
        return dp[0]