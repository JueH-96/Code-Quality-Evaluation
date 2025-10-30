class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        max_len = 1
        for i in range(n):
            inc_len = 1
            dec_len = 1
            for j in range(i + 1, n):
                if nums[j] > nums[j - 1]:
                    if dec_len == j - i: # still decreasing or just started
                        inc_len += 1
                    else:
                        break
                elif nums[j] < nums[j - 1]:
                    if inc_len == j - i: # still increasing or just started
                        dec_len += 1
                    else:
                        break
                else:
                    break
            max_len = max(max_len, inc_len, dec_len)
        return max_len