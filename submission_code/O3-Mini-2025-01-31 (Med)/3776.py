from typing import List
from functools import lru_cache

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        @lru_cache(maxsize=None)
        def dp(window, i):
            # window is a tuple representing the current front part of the list.
            # i is the index of the next element in nums to be appended (if any)
            
            # Fill the window to size 3 if possible (forced addition)
            if len(window) < 3 and i < n:
                # Forced: add the next element to the window (maintaining order)
                new_window = window + (nums[i],)
                return dp(new_window, i + 1)
            
            # If fewer than 3 elements remain and no more elements to fill them, we finish.
            if len(window) < 3:
                # When window is non-empty, the removal cost is the max of the remaining numbers.
                return max(window) if window else 0
            
            # Now we have exactly 3 elements in window.
            # We have three choices: remove any two of the three.
            # Each removal costs the maximum of the two removed,
            # and we keep the third element.
            # After that, we process the remaining list starting with the kept element.
            
            a, b, c = window[0], window[1], window[2]
            
            # Option 1: Remove a and b, keep c.
            cost1 = max(a, b) + dp((c,), i)
            # Option 2: Remove a and c, keep b.
            cost2 = max(a, c) + dp((b,), i)
            # Option 3: Remove b and c, keep a.
            cost3 = max(b, c) + dp((a,), i)
            
            return min(cost1, cost2, cost3)
        
        # Start with an empty window and index 0.
        return dp((), 0)