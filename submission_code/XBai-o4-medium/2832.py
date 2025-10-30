from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        groups = defaultdict(list)
        for i, num in enumerate(nums):
            groups[num].append(i)
        
        max_len = 0
        for g in groups.values():
            left = 0
            for right in range(len(g)):
                # Calculate the required deletions between left and right in the current group
                while (g[right] - right) - (g[left] - left) > k:
                    left += 1
                current_length = right - left + 1
                if current_length > max_len:
                    max_len = current_length
        return max_len