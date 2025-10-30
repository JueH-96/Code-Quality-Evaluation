class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_total_cost = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                cost = nums[0] + nums[i + 1] + nums[j + 1]
                min_total_cost = min(min_total_cost, cost)
        return min_total_cost