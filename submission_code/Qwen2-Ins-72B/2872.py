class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        max_val = 0
        for i in range(len(nums) - 1, -1, -1):
            if i == len(nums) - 1:
                max_val = nums[i]
            elif nums[i] <= max_val:
                max_val += nums[i]
            else:
                max_val = nums[i]
        return max_val