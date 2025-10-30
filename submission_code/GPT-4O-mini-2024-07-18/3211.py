class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize the maximum length of non-decreasing array
        max_length = 1
        
        # Start with the first element
        current_length = 1
        
        for i in range(1, n):
            if nums[i] >= nums[i - 1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        
        # Final check for the last segment
        max_length = max(max_length, current_length)
        
        return max_length