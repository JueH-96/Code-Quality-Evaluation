class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[:-3:-1]) + sum(nums[-3::-1])