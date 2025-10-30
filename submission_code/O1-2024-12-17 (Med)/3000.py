from typing import List
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # If x == 0, we just need the minimum difference between any two distinct elements.
        # That is found by sorting and taking the minimum gap between consecutive elements.
        if x == 0:
            nums_sorted = sorted(nums)
            ans = float('inf')
            for i in range(1, n):
                ans = min(ans, nums_sorted[i] - nums_sorted[i-1])
            return ans
        
        # For x > 0, use a balanced data structure (SortedList) to keep track
        # of all elements whose indices are at least x behind the current index i.
        # For each i, we look up the element nums[i] against the set of elements
        # from indices [0.. i-x].
        
        sl = SortedList()
        ans = float('inf')
        
        for i in range(n):
            # Add the element from index (i - x) if it exists.
            if i - x >= 0:
                sl.add(nums[i - x])
            
            if len(sl) > 0:
                # Find the closest element(s) in sl to nums[i] using bisect.
                w = nums[i]
                pos = sl.bisect_left(w)
                
                # Check the element at 'pos' (if pos is within range).
                if pos < len(sl):
                    ans = min(ans, abs(sl[pos] - w))
                
                # Also check the element just before 'pos' (if pos > 0).
                if pos > 0:
                    ans = min(ans, abs(sl[pos - 1] - w))
        
        return ans