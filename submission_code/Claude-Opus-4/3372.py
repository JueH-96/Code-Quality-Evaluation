class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        
        max_length = 1
        inc_length = 1  # Length of current increasing subarray
        dec_length = 1  # Length of current decreasing subarray
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                # Strictly increasing
                inc_length += 1
                dec_length = 1  # Reset decreasing length
            elif nums[i] < nums[i-1]:
                # Strictly decreasing
                dec_length += 1
                inc_length = 1  # Reset increasing length
            else:
                # Equal elements - reset both
                inc_length = 1
                dec_length = 1
            
            # Update max_length with the maximum of current lengths
            max_length = max(max_length, inc_length, dec_length)
        
        return max_length