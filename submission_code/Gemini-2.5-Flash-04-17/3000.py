from typing import List
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf') # Initialize minimum difference to infinity
        n = len(nums)
        
        # Use a sorted list to maintain elements nums[j] where j <= i - x.
        # As we iterate through the array with index i, we add nums[i - x] to this list
        # once i reaches x or more. This list will contain all nums[j] values
        # for indices j that are at least x positions behind the current index i.
        S = SortedList()

        # Iterate through the array with index i
        for i in range(n):
            # For the current index i, we are looking for an index j such that
            # abs(i - j) >= x. This condition means either j <= i - x or j >= i + x.
            # Our approach iterates through i from 0 to n-1.
            # We can handle the case j <= i - x by considering elements nums[j]
            # where j is in the range [0, i - x].

            # The element nums[i - x] becomes a candidate for the 'far' element
            # when its index (i - x) is non-negative. This happens when i >= x.
            if i >= x:
                # Add the element nums[i - x] to our sorted list S.
                # S now contains nums[j] for all j in the range [0, i - x].
                S.add(nums[i - x])

            # If S is not empty, it contains values nums[j] for indices j that satisfy j <= i - x.
            # Any such j satisfies i - j >= x. So, abs(i - j) = i - j >= x.
            # We find the minimum absolute difference between nums[i] and any element nums[j] where nums[j] is currently in S.
            
            # S will be non-empty if the condition `i >= x` was met at least once.
            # Since 0 <= x < n, the loop for i runs from 0 to n-1.
            # When i = x, the condition `i >= x` is true, and nums[0] is added to S.
            # Thus, S will have at least one element when i >= x.
            if S:
                target = nums[i]
                
                # Find the index where target would be inserted in S to maintain order.
                # bisect_left finds the index of the first element in S that is >= target.
                k_left = S.bisect_left(target)

                # Check the element at k_left (smallest element in S >= target) if k_left is a valid index.
                if k_left < len(S):
                    min_diff = min(min_diff, abs(target - S[k_left]))
                
                # Check the element at k_left - 1 (largest element in S < target) if k_left - 1 is a valid index (i.e., k_left > 0).
                if k_left > 0:
                    min_diff = min(min_diff, abs(target - S[k_left - 1]))
        
        # The loop considers all i from 0 to n-1. For each i where i >= x, it checks
        # pairs (i, j) where j <= i-x (i.e., i-j >= x). This covers all pairs (i, j)
        # such that i-j >= x.
        # By symmetry, if v-u >= x, the pair (v, u) will be considered when iterating i=v,
        # checking j=u (since u <= v-x). The absolute difference is the same.
        # Thus, all pairs (i, j) with abs(i-j) >= x are covered.

        return min_diff