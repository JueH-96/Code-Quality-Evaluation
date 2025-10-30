from typing import List
from collections import Counter

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        max_freq = 0
        left = 0
        total = 0

        for right in range(len(nums)):
            total += nums[right]

            while (right - left + 1) * nums[right] - total > numOperations * k:
                total -= nums[left]
                left += 1

            max_freq = max(max_freq, right - left + 1)

        return max_freq