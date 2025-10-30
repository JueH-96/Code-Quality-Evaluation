class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_value = nums[-1]
        
        for i in range(n - 2, -1, -1):
            if nums[i] <= nums[i + 1]:
                nums[i] += nums[i + 1]
            max_value = max(max_value, nums[i])
        
        return max_value