from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        index_map = defaultdict(list)
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        max_length = 0
        
        for indices in index_map.values():
            left = 0
            for right in range(1, len(indices)):
                # Calculate the number of deletions needed
                deletions = indices[right] - indices[left] - (right - left)
                while deletions > k:
                    left += 1
                    deletions = indices[right] - indices[left] - (right - left)
                max_length = max(max_length, right - left + 1)
        
        return max_length