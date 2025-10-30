class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        # longest increasing and decreasing subarray ending at i
        inc, dec = [1]*n, [1]*n
        
        # filling the longest increasing subarray
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                inc[i] = inc[i-1] + 1
        
        # filling the longest decreasing subarray
        for i in range(n-2, -1, -1):
            if nums[i] > nums[i+1]:
                dec[i] = dec[i+1] + 1
        
        # Getting the maximum value between inc and dec
        return max(max(inc), max(dec))