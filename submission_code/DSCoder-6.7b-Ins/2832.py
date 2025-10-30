from typing import List
from collections import deque

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        num_counts = [0] * (n + 1)
        max_count = 0
        left = 0
        for right in range(n):
            max_count = max(max_count, num_counts[nums[right]])
            if right - left + 1 > max_count + k:
                num_counts[nums[left]] -= 1
                left += 1
            num_counts[nums[right]] += 1
        return max_count + k