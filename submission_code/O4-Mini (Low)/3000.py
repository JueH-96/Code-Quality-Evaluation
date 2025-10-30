from bisect import bisect_left, insort
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # We maintain a sorted list of all nums[i] for i <= j - x
        # As we sweep j from x to n-1, we insert nums[j-x] into the sorted list,
        # then we query the closest values to nums[j] in that list.
        
        n = len(nums)
        sorted_vals = []
        ans = float('inf')
        
        for j in range(x, n):
            # Insert the element that is now at least x away from j
            # i = j - x
            val_to_add = nums[j - x]
            insort(sorted_vals, val_to_add)
            
            # Now find where nums[j] would be inserted
            v = nums[j]
            pos = bisect_left(sorted_vals, v)
            
            # Check the neighbor at pos
            if pos < len(sorted_vals):
                ans = min(ans, abs(sorted_vals[pos] - v))
                if ans == 0:
                    return 0
            # Check the neighbor before pos
            if pos > 0:
                ans = min(ans, abs(sorted_vals[pos - 1] - v))
                if ans == 0:
                    return 0
        
        return ans