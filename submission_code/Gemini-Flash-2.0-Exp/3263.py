class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cost1 = nums[0]
                cost2 = nums[i]
                cost3 = nums[j]
                min_cost = min(min_cost, cost1 + cost2 + cost3)
        return min_cost