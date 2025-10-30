import collections
from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # If the array has 0 or 1 element, no operations can be performed
        # because we need at least two elements (i < j) for an operation.
        # The length remains as is.
        if n <= 1:
            return n

        # Initialize the left pointer for the first half of elements.
        i = 0
        # Initialize the right pointer for the second half of elements.
        # Starting j at n // 2 ensures i < j for any potential pair,
        # as i will always be less than n // 2.
        j = n // 2
        
        # Counter for the number of successful pairs removed.
        count = 0
        
        # Iterate as long as both pointers are within their logical bounds.
        # 'i' iterates through the first half (up to n//2 - 1).
        # 'j' iterates through the second half (from n//2 to n - 1).
        while i < n // 2 and j < n:
            # Check if the current elements form a valid pair (nums[i] < nums[j]).
            if nums[i] < nums[j]:
                # If valid, we perform the removal (conceptually).
                # Increment the count of removed pairs.
                count += 1
                # Advance both pointers, as these elements are now "gone".
                i += 1
                j += 1
            else:
                # If nums[i] >= nums[j], this particular nums[j] is too small
                # to be paired with nums[i] (or any subsequent nums[k] where k > i,
                # because nums[k] will be >= nums[i]).
                # So, we advance 'j' to look for a larger element in the second half.
                # 'i' remains in place, waiting for a suitable match.
                j += 1
        
        # The minimum length is the original length minus twice the number of pairs removed.
        return n - (2 * count)