from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0

        while i < n:
            # Count the length of the current alternating subarray
            length = 1
            while i + 1 < n and nums[i] != nums[i + 1]:
                i += 1
                length += 1

            # Every subarray of an alternating subarray is also alternating
            count += (length * (length + 1)) // 2
            i += 1

        return count