class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        nums.sort()
        while len(nums) > 1:
            nums[-2] += nums[-1]
            nums.pop()
        return nums[0]