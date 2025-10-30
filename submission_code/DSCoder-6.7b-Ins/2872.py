class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = nums[-1]
        for i in range(n-2, -1, -1):
            if nums[i] <= max_val:
                max_val = nums[i] + max_val
            else:
                max_val = nums[i]
        return max_val