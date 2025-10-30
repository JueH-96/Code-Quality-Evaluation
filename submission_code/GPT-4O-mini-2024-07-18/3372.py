class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_length = 1
        current_length = 1
        
        # Check for strictly increasing and decreasing subarrays
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:  # strictly increasing
                current_length += 1
            elif nums[i] < nums[i - 1]:  # strictly decreasing
                current_length += 1
            else:  # reset on equal elements
                max_length = max(max_length, current_length)
                current_length = 1
            
            # Update max_length for the last segment
            max_length = max(max_length, current_length)
        
        return max_length