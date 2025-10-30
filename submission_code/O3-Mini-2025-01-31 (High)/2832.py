from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Dictionary mapping each number to its list of positions.
        pos_map = defaultdict(list)
        for i, num in enumerate(nums):
            pos_map[num].append(i)
            
        max_length = 0
        
        # For each unique number, use a sliding window over its positions.
        for num, indices in pos_map.items():
            left = 0
            # Iterate over positions with 'right' pointer.
            for right in range(len(indices)):
                # The cost to make the subarray (from indices[left] to indices[right])
                # equal is determined by the gap between the indices minus the count of that number that we already have.
                # That is: cost = (indices[right] - indices[left] + 1) - (right - left + 1)
                # Simplified: cost = indices[right] - indices[left] - (right - left)
                while indices[right] - indices[left] - (right - left) > k:
                    left += 1
                # Update maximum length of subarray for this number.
                max_length = max(max_length, right - left + 1)
        
        return max_length