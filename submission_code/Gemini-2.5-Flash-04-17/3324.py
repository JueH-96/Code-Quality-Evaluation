from typing import List
from collections import Counter

# Delimiter
class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        """
        Checks if an array of even length can be split into two arrays of equal length
        such that both contain distinct elements.

        Args:
            nums: An integer array of even length.

        Returns:
            True if a split is possible, False otherwise.
        """
        # Count the frequency of each number in the input array.
        # We can use collections.Counter for this.
        counts = Counter(nums)

        # Iterate through the counts. If the frequency of any number
        # is greater than 2, it's impossible to form two subarrays
        # where each contains distinct elements. This is because each
        # subarray can contain at most one instance of any given number.
        # Therefore, between the two subarrays combined, a number can
        # appear at most 1 (in nums1) + 1 (in nums2) = 2 times in total.
        # If any number appears more than twice in the original array,
        # there is no way to distribute its occurrences without having
        # duplicates in either nums1 or nums2.
        for count in counts.values():
            if count > 2:
                return False

        # If we iterate through all numbers and none have a frequency
        # greater than 2, it means every number appears at most twice.
        # In this case, it is always possible to form the two subarrays
        # as required. For any number that appears twice, place one instance
        # in nums1 and the other in nums2. For any number that appears once,
        # it can be placed in either nums1 or nums2. Since the total number
        # of single-occurrence elements matches the total number of remaining
        # slots in nums1 and nums2, and all single-occurrence elements are
        # distinct from each other and from the double-occurrence elements,
        # we can distribute them arbitrarily to fill the remaining slots while
        # maintaining the distinctness property in both subarrays.
        return True
# Delimiter