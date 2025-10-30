from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        frequency = defaultdict(int)

        for right in range(len(nums)):
            frequency[nums[right]] += 1

            while frequency[nums[right]] > k:
                frequency[nums[left]] -= 1
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length