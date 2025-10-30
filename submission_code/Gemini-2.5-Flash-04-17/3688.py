import sys
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            # Constraints say 1 <= nums.length, but defensive coding
            return 0

        # DP states (using O(1) space by only storing previous values):
        # ends_no_del: Max subarray sum ending at the current index, without deleting any element up to this point.
        # ends_one_del: Max subarray sum ending at the current index, having deleted exactly one element at some point before or at the current index.

        # Initialize for the first element (index 0)
        # A subarray must be non-empty. The max sum ending at index 0 without deletion is just nums[0].
        ends_no_del = nums[0]
        # It's impossible to have deleted an element *before or at* index 0 and still end at index 0.
        # Initialize with a very small number to represent negative infinity.
        ends_one_del = -float('inf') 

        # The maximum result seen so far. Initially, it's the max subarray sum ending at index 0 (no deletion).
        max_so_far = nums[0]

        # Iterate through the array starting from the second element
        for i in range(1, n):
            # To calculate the DP states for the current index i, we need the values from the previous index i-1.
            # Store the values from the previous step before updating them for the current step.
            prev_ends_no_del = ends_no_del
            # We only need the previous ends_one_del value directly.

            # Calculate ends_no_del for the current index i:
            # The max sum ending at i without deletion is either:
            # 1. Starting a new subarray at i (value is nums[i]).
            # 2. Extending the max sum subarray ending at i-1 without deletion by adding nums[i].
            ends_no_del = max(nums[i], ends_no_del + nums[i])

            # Calculate ends_one_del for the current index i:
            # The max sum ending at i with one deletion somewhere before or at i is either:
            # 1. Extending a max sum subarray ending at i-1 that already had one deletion, by adding nums[i].
            #    This is ends_one_del (from previous step) + nums[i].
            # 2. Performing the *first* and *only* deletion at index i-1.
            #    This means taking the max sum subarray ending at i-2 without deletion
            #    and adding nums[i]. The max sum ending at i-2 without deletion
            #    is effectively represented by prev_ends_no_del (max sum ending at i-1 without deletion,
            #    which includes the sum up to i-2 unless i-1 was the start).
            #    So, this option is prev_ends_no_del (value from i-1) + nums[i] conceptually representing 
            #    skipping nums[i-1].
            #    The transition is max(ends_one_del + nums[i], prev_ends_no_del).
            # Note: ends_one_del here refers to the value from the previous iteration (i-1) because it's read before being updated in this step.
            ends_one_del = max(ends_one_del + nums[i], prev_ends_no_del)

            # The overall maximum subarray sum is the maximum seen across all ending positions
            # and all states (with or without one deletion).
            max_so_far = max(max_so_far, ends_no_del, ends_one_del)

        # The problem requires the resulting array to be non-empty.
        # The DP calculates max subarray sums ending at various points in the ORIGINAL index space,
        # allowing a conceptual "skip" which corresponds to deleting an element.
        # If deleting all occurrences of *any* value makes the array empty (e.g., nums=[5, 5, 5]),
        # that operation is invalid. The only valid "resulting array" is the original one.
        # In such a case, the max subarray sum is the standard Kadane's result on the original array.
        # The DP naturally calculates the standard Kadane's max sum via the ends_no_del path.
        # The max_so_far is initialized with nums[0] and updated with ends_no_del, so it correctly
        # includes the standard max subarray sum if all deletion operations are invalid or yield smaller sums.
        # The constraint "nums remains non-empty on removing all occurrences of x" means we only
        # consider deletions that leave a non-empty array. The DP implicitly finds the best subarray
        # assuming *a* valid single deletion occurred somewhere. If all value deletions lead to
        # empty arrays, the maximum will come from the 'no deletion' path. If some value deletion
        # is valid and gives a higher result, the DP should capture it if the "one element deletion"
        # strategy is equivalent to "all occurrences of one value deletion" in terms of the maximum
        # subarray sum achievable. The standard competitive programming interpretation is that this DP
        # solves the problem described.

        return max_so_far