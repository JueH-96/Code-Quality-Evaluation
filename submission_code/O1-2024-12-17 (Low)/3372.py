class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # We'll track two running lengths:
        # inc_len for the current strictly increasing subarray
        # dec_len for the current strictly decreasing subarray
        inc_len = 1
        dec_len = 1
        result = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                inc_len += 1
                dec_len = 1
            elif nums[i] < nums[i - 1]:
                dec_len += 1
                inc_len = 1
            else:
                # Reset both if consecutive elements are equal
                inc_len = 1
                dec_len = 1
            
            # Update the global maximum
            result = max(result, inc_len, dec_len)
        
        return result