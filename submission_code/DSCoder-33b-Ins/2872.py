class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        nums.reverse()
        max_val = nums[0]
        for i in range(1, len(nums)):
            if nums[i] >= max_val:
                max_val = nums[i]
            else:
                max_val = max(max_val, nums[i-1] + nums[i])
        return max_val