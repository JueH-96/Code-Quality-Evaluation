class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_len = 1
        n = len(nums)
        # Check for strictly increasing
        inc_len = 1
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                inc_len += 1
                if inc_len > max_len:
                    max_len = inc_len
            else:
                inc_len = 1
        # Check for strictly decreasing
        dec_len = 1
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                dec_len += 1
                if dec_len > max_len:
                    max_len = dec_len
            else:
                dec_len = 1
        return max_len