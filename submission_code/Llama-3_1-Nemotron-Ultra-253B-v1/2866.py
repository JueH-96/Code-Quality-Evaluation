from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_r = l
                for r in range(l + 1, n):
                    if nums[r] > threshold:
                        break
                    if (nums[r] % 2) == (nums[r - 1] % 2):
                        break
                    current_r = r
                current_length = current_r - l + 1
                if current_length > max_length:
                    max_length = current_length
        return max_length