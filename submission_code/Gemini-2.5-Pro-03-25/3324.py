import collections
from typing import List

class Solution:
    """
    Solves the problem of determining if an array can be split into two subarrays
    of equal length, each containing distinct elements.
    """
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        """
        Checks if the given array 'nums' can be split into two subarrays
        nums1 and nums2 such that:
        - nums1.length == nums2.length == nums.length / 2
        - nums1 contains distinct elements.
        - nums2 contains distinct elements.

        Args:
            nums: A list of integers with even length.

        Returns:
            True if such a split is possible, False otherwise.
        """

        # The core idea is based on the frequency of each number.
        # If we need to split nums into two subarrays (nums1, nums2) with distinct elements,
        # any particular number 'x' can appear at most once in nums1 and at most once in nums2.
        # Therefore, the total count of any number 'x' in the original array 'nums'
        # cannot exceed 2. If any number appears 3 or more times, it's impossible
        # to satisfy the distinctness condition for both subarrays.

        # Calculate the frequency of each number in the input array.
        # collections.Counter is an efficient way to do this.
        counts = collections.Counter(nums)

        # Iterate through the frequencies of each unique number found.
        for num in counts:
            # If the frequency of any number is greater than 2,
            # it's impossible to place all occurrences of this number
            # into nums1 and nums2 while keeping them distinct.
            if counts[num] > 2:
                return False

        # If we have iterated through all unique numbers and none
        # had a frequency greater than 2, then a valid split is possible.
        # We don't need to construct the actual split, just determine possibility.
        return True