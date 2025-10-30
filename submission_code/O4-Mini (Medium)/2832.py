from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Record the indices where each value appears
        positions = defaultdict(list)
        for idx, val in enumerate(nums):
            positions[val].append(idx)
        
        ans = 0
        # For each value, use a sliding window on its index list
        for pos_list in positions.values():
            left = 0
            for right in range(len(pos_list)):
                # Total elements in nums between pos_list[left]..pos_list[right] is
                # span = pos_list[right] - pos_list[left] + 1
                # Number of target-value elements in that span is (right-left+1)
                # So non-targets = span - (right-left+1). Must be <= k.
                while left <= right and (pos_list[right] - pos_list[left] + 1) - (right - left + 1) > k:
                    left += 1
                # After deleting the non-targets, we get (right-left+1) equal elements
                ans = max(ans, right - left + 1)
        
        return ans