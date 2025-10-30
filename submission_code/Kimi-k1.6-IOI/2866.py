from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_len = 1
                max_len = max(max_len, current_len)
                r = l + 1
                while r < n:
                    if nums[r] > threshold:
                        break
                    if (nums[r] % 2) == (nums[r-1] % 2):
                        break
                    current_len += 1
                    max_len = max(max_len, current_len)
                    r += 1
        return max_len