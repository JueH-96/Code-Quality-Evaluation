from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        i = 0

        while i < nums.length - 1:
            if nums[i + 1] == nums[i] + 1:
                start = i
                length = 2
                i += 2

                while i < nums.length and nums[i] == nums[start + (i - start) % 2]:
                    length += 1
                    i += 1

                max_len = max(max_len, length)
            else:
                i += 1

        return max_len