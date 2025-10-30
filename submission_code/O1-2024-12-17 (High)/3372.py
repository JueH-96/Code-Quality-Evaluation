class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # At minimum, the longest strictly monotonic subarray has length 1
        max_len = 1
        inc_len = 1  # length of current strictly increasing subarray
        dec_len = 1  # length of current strictly decreasing subarray
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                inc_len += 1
                dec_len = 1
            elif nums[i] < nums[i - 1]:
                dec_len += 1
                inc_len = 1
            else:
                # Equal case resets both because we need strict inequality
                inc_len = 1
                dec_len = 1
            
            max_len = max(max_len, inc_len, dec_len)
        
        return max_len