class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1
        
        # Initialize the length of the longest non-decreasing subarray
        max_length = 1
        current_length = 1
        
        for i in range(1, n):
            if nums[i] >= nums[i - 1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        
        max_length = max(max_length, current_length)
        
        return max_length