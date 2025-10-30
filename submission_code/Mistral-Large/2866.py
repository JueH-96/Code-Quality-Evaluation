from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        current_len = 0
        i = 0

        while i < nums.length:
            if nums[i] <= threshold and nums[i] % 2 == 0:
                current_len = 1
                j = i + 1
                while j < nums.length and nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                    current_len += 1
                    j += 1
                max_len = max(max_len, current_len)
                i = j
            else:
                i += 1

        return max_len