class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        nums.sort()
        return min(nums[0]+nums[3], nums[1]+nums[2], nums[0]+nums[1], nums[2]+nums[3])