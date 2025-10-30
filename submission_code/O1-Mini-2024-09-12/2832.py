from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        num_positions = defaultdict(list)
        for idx, num in enumerate(nums):
            num_positions[num].append(idx)
        
        max_length = 0
        
        for num, positions in num_positions.items():
            left = 0
            for right in range(len(positions)):
                # Calculate deletions required to make positions[left:right+1] contiguous
                deletions = positions[right] - positions[left] + 1 - (right - left + 1)
                while deletions > k:
                    left += 1
                    deletions = positions[right] - positions[left] + 1 - (right - left + 1)
                current_length = right - left + 1
                if current_length > max_length:
                    max_length = current_length
        
        return max_length