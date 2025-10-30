class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0

        # Calculate the prefix sum array
        prefix_sum = [0] * n
        prefix_sum[0] = nums[0]
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i]

        # Calculate the cost for each possible split point
        max_cost = float('-inf')
        for i in range(n - 1):
            cost1 = prefix_sum[i] + (i + 1) % 2 * nums[i]
            cost2 = prefix_sum[n - 1] - prefix_sum[i] + (n - i - 1) % 2 * nums[i + 1]
            max_cost = max(max_cost, cost1 + cost2)

        return max_cost