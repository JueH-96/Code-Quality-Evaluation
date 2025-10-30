from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        indices = defaultdict(list)
        left = 0
        max_len = 0
        for i, x in enumerate(nums):
            indices[x].append(i)
            if len(indices[x]) > k:
                # Calculate the required left boundary to exclude the earliest occurrence
                # that causes the current element to exceed k occurrences
                required_left = indices[x][len(indices[x]) - k - 1] + 1
                left = max(left, required_left)
            current_len = i - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len