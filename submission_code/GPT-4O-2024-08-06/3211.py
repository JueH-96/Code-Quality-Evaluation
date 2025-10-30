class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # The problem boils down to finding the longest non-decreasing subarray
        # because we can always replace the rest of the array with its sum.
        
        # Initialize variables to track the maximum length of non-decreasing subarray
        max_length = 1
        current_length = 1
        
        # Iterate through the array to find the longest non-decreasing subarray
        for i in range(1, len(nums)):
            if nums[i] >= nums[i - 1]:
                current_length += 1
            else:
                current_length = 1
            
            # Update the maximum length found so far
            max_length = max(max_length, current_length)
        
        return max_length