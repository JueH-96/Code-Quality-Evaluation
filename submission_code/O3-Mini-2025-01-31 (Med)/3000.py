from bisect import bisect_left, insort
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize answer with a large number.
        ans = float('inf')
        
        # We'll use a sorted list (BST emulation) to store candidate values.
        sorted_list = []
        
        # There are two cases:
        # 1. When x == 0, every pair of distinct indices is allowed.
        #    So for index i, candidate set is indices [0, i).
        # 2. When x > 0, for index i we are allowed candidate indices [0, i - x].
        if x == 0:
            # For i=0, candidate set is empty.
            for i in range(n):
                # For index i, the candidate set is all indices < i (already in sorted_list)
                if sorted_list:
                    pos = bisect_left(sorted_list, nums[i])
                    if pos < len(sorted_list):
                        ans = min(ans, abs(sorted_list[pos] - nums[i]))
                    if pos > 0:
                        ans = min(ans, abs(sorted_list[pos-1] - nums[i]))
                # Insert current element into sorted_list while maintaining sorted order.
                insort(sorted_list, nums[i])
        else:
            # For x > 0, for a given index i, the allowed candidate indices are those j for which
            # j <= i - x. We add candidate numbers as soon as they become available.
            for i in range(n):
                # When i - x >= 0, candidate at index (i - x) becomes available.
                if i - x >= 0:
                    insort(sorted_list, nums[i-x])
                # Only do query if we have at least one candidate.
                if sorted_list:
                    pos = bisect_left(sorted_list, nums[i])
                    if pos < len(sorted_list):
                        ans = min(ans, abs(sorted_list[pos] - nums[i]))
                    if pos > 0:
                        ans = min(ans, abs(sorted_list[pos-1] - nums[i]))
        # Return answer as an integer.
        return ans