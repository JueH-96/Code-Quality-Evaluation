from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        max_freq = max_length = 0
        left = 0

        for right in range(len(nums)):
            count[nums[right]] += 1
            max_freq = max(max_freq, count[nums[right]])

            if right - left + 1 - max_freq > k:
                count[nums[left]] -= 1
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length