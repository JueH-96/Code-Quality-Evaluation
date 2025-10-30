class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        
        max_length = 1
        increasing = 1
        decreasing = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                increasing += 1
                decreasing = 1
            elif nums[i] < nums[i-1]:
                decreasing += 1
                increasing = 1
            else:
                increasing = 1
                decreasing = 1
            
            max_length = max(max_length, increasing, decreasing)
        
        return max_length