import sys 
# The default recursion depth limit in Python might be insufficient for 
# very large inputs if a recursive data structure like a Segment Tree were used.
# It's generally not needed for iterative approaches or libraries like sortedcontainers.
# sys.setrecursionlimit(2000) 

from typing import List
import math 

# Attempt to import SortedList from the 'sortedcontainers' library.
# This library provides an efficient implementation of a sorted list
# with O(log N) time complexity for additions, deletions, and searches.
try:
    from sortedcontainers import SortedList
# If the 'sortedcontainers' library is not installed or available, 
# define a fallback class using Python's built-in list and bisect module.
# Important Note: This fallback implementation has O(N) complexity for the 'add' operation 
# due to list insertion, which makes the overall algorithm O(N^2). 
# The problem constraints (N <= 10^5) suggest an O(N log N) solution is required,
# which the library version provides. This fallback is for completeness but may TLE.
except ImportError:
    import bisect
    class SortedList:
        """
        A basic fallback sorted list implementation using Python's list and bisect.
        WARNING: `add` operation is O(N), making the overall solution potentially O(N^2).
        """
        def __init__(self, iterable=None):
            # Initialize with a sorted list from the iterable, if provided.
            self._list = sorted(list(iterable)) if iterable is not None else []

        def add(self, value):
            """Inserts value while maintaining sorted order. O(N) complexity."""
            # bisect.insort uses binary search to find position but list insertion takes O(N).
            bisect.insort(self._list, value)

        def bisect_left(self, value):
            """Finds insertion point for value using binary search. O(log N) complexity."""
            return bisect.bisect_left(self._list, value)
        
        def __len__(self):
            """Returns the number of items in the list. O(1) complexity."""
            return len(self._list)

        def __getitem__(self, index):
            """Gets item at the specified index. O(1) complexity."""
            # Python's list access raises IndexError if index is out of bounds.
            return self._list[index]


class Solution:
    """
    Implements the solution to find the minimum absolute difference between two elements
    in the input array `nums` such that their indices are at least `x` positions apart.
    """
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        """
        Calculates the minimum absolute difference between two elements nums[i] and nums[j]
        such that abs(i - j) >= x.

        Args:
            nums: A list of integers.
            x: The minimum required distance between the indices of the two elements.

        Returns:
            The minimum absolute difference found that satisfies the index distance constraint.
            Returns 0 if x is 0.
        """
        
        n = len(nums)
        
        # Handle the edge case where x is 0.
        # If x is 0, the condition abs(i - j) >= 0 is always true for any pair of indices (i, j).
        # We can choose i = j, which satisfies the condition and gives an absolute difference
        # of abs(nums[i] - nums[j]) = 0. This is the minimum possible absolute difference.
        if x == 0:
            return 0

        # Initialize the minimum difference found so far to positive infinity.
        min_diff = float('inf')
        
        # Use a SortedList data structure to efficiently maintain and query elements.
        # The 'window' will store elements nums[k] such that k <= i - x, kept in sorted order.
        # This allows us to quickly find elements close to nums[i] among the valid preceding elements.
        window = SortedList()
        
        # Iterate through the array starting from index x.
        # For each index `i`, `nums[i]` is considered as the potential right element of a valid pair (j, i).
        # The potential left element `nums[j]` must satisfy `j <= i - x`.
        for i in range(x, n):
            # Add the element `nums[i - x]` to the sorted window.
            # This element `nums[i-x]` is exactly `x` indices before the current element `nums[i]`.
            # Therefore, the pair `(i - x, i)` satisfies the condition `abs(i - (i - x)) = x >= x`.
            # All elements already in the window also satisfy the condition with `nums[i]`.
            # Using SortedList's add method (O(log N) if using sortedcontainers).
            window.add(nums[i - x])
            
            # Find the insertion point `idx` for the current element `nums[i]` in the sorted `window`.
            # `bisect_left` returns the index where `nums[i]` would be inserted to maintain order.
            # This means `window[idx]` (if `idx` is within bounds) is the smallest element in `window` >= `nums[i]`.
            # And `window[idx - 1]` (if `idx > 0`) is the largest element in `window` < `nums[i]`.
            idx = window.bisect_left(nums[i])
            
            # Check the element at the insertion point `idx` (the potential successor or equal value).
            # This element `window[idx]` is the closest element in the window that is >= `nums[i]`.
            if idx < len(window):
                # Calculate the absolute difference and update `min_diff` if this difference is smaller.
                diff = abs(nums[i] - window[idx])
                min_diff = min(min_diff, diff)
            
            # Check the element just before the insertion point `idx` (the potential predecessor value).
            # This element `window[idx - 1]` is the closest element in the window that is < `nums[i]`.
            # The condition `idx > 0` ensures that `idx - 1` is a valid non-negative index.
            if idx > 0:
                # Calculate the absolute difference and update `min_diff` if smaller.
                diff = abs(nums[i] - window[idx - 1])
                min_diff = min(min_diff, diff)

            # Optimization: If the minimum difference found is 0, it's the smallest possible value.
            # We can stop the search early and return 0.
            if min_diff == 0:
                return 0
                
        # After iterating through all relevant indices `i`, `min_diff` holds the minimum absolute
        # difference found between any pair `(nums[j], nums[i])` such that `abs(i - j) >= x`.
        # The problem constraints (1 <= n, 0 <= x < n) ensure that the loop runs at least once 
        # for n >= 1 (unless n=1, x=0 case handled earlier), so min_diff will be updated from infinity.
        return min_diff