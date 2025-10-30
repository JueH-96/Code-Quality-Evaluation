from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_length = 0
        for num in set(nums):
            left = 0
            count = 0
            for right in range(len(nums)):
                if nums[right] != num:
                    count += 1
                while count > k:
                    if nums[left] != num:
                        count -= 1
                    left += 1
                max_length = max(max_length, right - left + 1)
        return max_length