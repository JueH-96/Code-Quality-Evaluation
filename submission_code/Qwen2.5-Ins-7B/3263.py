class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        return nums[0] + min(nums[1], nums[2], nums[3]) + min(nums[4:], default=nums[1])