class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        
        max_length = 1
        increasing_length = 1
        decreasing_length = 1
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                increasing_length += 1
                decreasing_length = 1
            elif nums[i] < nums[i - 1]:
                decreasing_length += 1
                increasing_length = 1
            else:
                increasing_length = 1
                decreasing_length = 1
            
            max_length = max(max_length, increasing_length, decreasing_length)
        
        return max_length