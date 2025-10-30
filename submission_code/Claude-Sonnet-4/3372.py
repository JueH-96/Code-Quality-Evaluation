class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        
        max_length = 1
        inc_length = 1  # length of current strictly increasing subarray
        dec_length = 1  # length of current strictly decreasing subarray
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                # Strictly increasing
                inc_length += 1
                dec_length = 1
            elif nums[i] < nums[i-1]:
                # Strictly decreasing
                dec_length += 1
                inc_length = 1
            else:
                # Equal elements - reset both
                inc_length = 1
                dec_length = 1
            
            max_length = max(max_length, inc_length, dec_length)
        
        return max_length