class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        i = len(nums) - 2
        max_val = nums[-1]
        
        while i >= 0:
            if nums[i] <= nums[i + 1]:
                nums[i + 1] += nums[i]
            max_val = max(max_val, nums[i + 1])
            i -= 1
        
        return max_val