class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        # Find the longest contiguous subarray of the same value
        max_len = 1
        current_len = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                current_len += 1
            else:
                max_len = max(max_len, current_len)
                current_len = 1
        # Also check the wrap-around case
        if nums[0] == nums[-1]:
            i = 0
            while i < n and nums[i] == nums[-1]:
                current_len += 1
                i += 1
        max_len = max(max_len, current_len)
        
        # Calculate the minimum seconds needed
        # If max_len is n, all elements are already equal
        if max_len == n:
            return 0
        
        # The minimum number of steps required is the ceiling of log_2(n/max_len)
        from math import ceil, log2
        return ceil(log2(n / max_len))