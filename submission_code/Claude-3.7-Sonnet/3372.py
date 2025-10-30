class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        if n == 1:
            return 1
        
        inc_len = 1  # Length of current increasing subarray
        dec_len = 1  # Length of current decreasing subarray
        max_len = 1  # Maximum length of monotonic subarray
        
        for i in range(1, n):
            if nums[i] > nums[i-1]:  # Current element is greater than previous
                inc_len += 1
                dec_len = 1  # Reset decreasing counter
            elif nums[i] < nums[i-1]:  # Current element is less than previous
                dec_len += 1
                inc_len = 1  # Reset increasing counter
            else:  # Current element equals previous (neither increasing nor decreasing)
                inc_len = 1
                dec_len = 1
            
            max_len = max(max_len, inc_len, dec_len)
        
        return max_len