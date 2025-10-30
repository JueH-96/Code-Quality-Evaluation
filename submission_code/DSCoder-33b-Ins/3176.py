class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        nums.sort()
        return nums[0] + nums[1] + nums[2]