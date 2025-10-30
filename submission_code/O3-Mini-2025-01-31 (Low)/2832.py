from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Group indices for each distinct value in nums
        groups = defaultdict(list)
        for idx, num in enumerate(nums):
            groups[num].append(idx)
        
        max_length = 0
        
        # For each group of equal elements (based on value), try sliding window over indices.
        for indices in groups.values():
            left = 0
            # We want to maximize count = right - left + 1 such that cost <= k.
            # cost = (indices[right] - indices[left]) - (window length - 1)
            for right in range(len(indices)):
                # While condition not satisfied, slide left pointer.
                while left <= right and (indices[right] - indices[left] - (right - left)) > k:
                    left += 1
                # Window length is (right - left + 1)
                max_length = max(max_length, right - left + 1)
        
        return max_length