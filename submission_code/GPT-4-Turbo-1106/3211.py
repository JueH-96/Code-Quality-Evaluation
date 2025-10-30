class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # Initialize the length of the longest non-decreasing subarray
        max_length = 1
        # Initialize the current length of the non-decreasing subarray
        current_length = 1
        
        # Iterate through the array to find the longest non-decreasing subarray
        for i in range(1, len(nums)):
            # If the current element is greater than or equal to the previous one
            if nums[i] >= nums[i - 1]:
                # Increase the current length
                current_length += 1
                # Update the maximum length if necessary
                max_length = max(max_length, current_length)
            else:
                # Reset the current length to 1
                current_length = 1
        
        return max_length