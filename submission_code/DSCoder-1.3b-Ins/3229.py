class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        total_cost = 0
        for i in range(len(nums)):
            if i % 3 == 2:
                total_cost += nums[i]
        return total_cost