from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        num_indices = defaultdict(list)
        for i, num in enumerate(nums):
            num_indices[num].append(i)
        
        max_length = 0
        
        for indices in num_indices.values():
            left = 0
            for right in range(len(indices)):
                while (indices[right] - indices[left] + 1) - (right - left + 1) > k:
                    left += 1
                max_length = max(max_length, right - left + 1)
        
        return max_length