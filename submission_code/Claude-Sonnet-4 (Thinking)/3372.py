class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        
        max_len = 1
        inc_len = 1  # length of strictly increasing subarray ending at current position
        dec_len = 1  # length of strictly decreasing subarray ending at current position
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                inc_len += 1
                dec_len = 1
            elif nums[i] < nums[i-1]:
                dec_len += 1
                inc_len = 1
            else:  # nums[i] == nums[i-1]
                inc_len = 1
                dec_len = 1
            
            max_len = max(max_len, inc_len, dec_len)
        
        return max_len