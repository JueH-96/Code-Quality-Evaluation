from typing import List
# sortedcontainers is a third-party library. It's commonly available in competitive
# programming environments. If not, a balanced BST or segment tree would be needed
# to achieve the same O(N log N) complexity.
from sortedcontainers import SortedList

class Solution:
  def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
    n = len(nums)

    # If x is 0, we are allowed to choose i=j.
    # In this case, abs(i-j) = 0, which satisfies abs(i-j) >= x (0 >= 0).
    # The difference abs(nums[i] - nums[j]) would be abs(nums[i] - nums[i]) = 0.
    # Since absolute difference cannot be negative, 0 is the minimum possible.
    if x == 0:
        return 0

    min_diff = float('inf')
    
    # 'sl' will maintain a sorted list of elements nums[k] where k <= j-x.
    # These are the "candidate" elements from the "left" part of the array,
    # sufficiently far from the current nums[j].
    sl = SortedList()

    # Iterate j from x to n-1. nums[j] is the "right" element of a potential pair.
    # For each nums[j], elements nums[0...j-x] are valid "left" elements.
    # The element nums[j-x] is the one that just became available as a "left" candidate
    # as we advance j.
    for j in range(x, n):
        # Add nums[j-x] to the sorted list of available "left" candidates.
        # This operation takes O(log K) where K is the current size of sl.
        sl.add(nums[j-x])
        
        current_val = nums[j]
        
        # We need to find an element in 'sl' that is closest to 'current_val'.
        # In a sorted list, such an element must be either the predecessor or successor
        # of 'current_val' (or 'current_val' itself if present).
        
        # Find the insertion point for current_val in sl using binary search.
        # bisect_left returns an index 'idx' such that all sl[k] for k < idx
        # have sl[k] < current_val, and all sl[k] for k >= idx
        # have sl[k] >= current_val. This is O(log K).
        idx = sl.bisect_left(current_val)
        
        # Candidate 1: Check sl[idx]
        # This is the smallest element in sl that is >= current_val (successor or equal).
        # This check is valid if idx is within the bounds of sl (i.e., idx < len(sl)).
        if idx < len(sl):
            diff = abs(sl[idx] - current_val)
            min_diff = min(min_diff, diff)
        
        # Candidate 2: Check sl[idx-1]
        # This is the largest element in sl that is < current_val (predecessor).
        # This check is valid if idx > 0 (meaning there's an element to its left).
        if idx > 0:
            # sl[idx-1] is nums[k] where nums[k] < current_val
            diff = abs(sl[idx-1] - current_val)
            min_diff = min(min_diff, diff)
        
        # Optimization: if min_diff becomes 0, it's the smallest possible absolute difference.
        # We can stop early.
        if min_diff == 0:
            return 0
            
    return min_diff