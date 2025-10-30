import collections
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        """
        Determines if the array can be split into two subarrays with distinct elements.

        The core logic is based on the frequency of elements. For a valid split
        into two arrays, nums1 and nums2, where both have distinct elements, any
        given number can appear at most once in nums1 and at most once in nums2.
        This implies that the total count of any number in the original `nums`
        array cannot be greater than 2. If a number appears 3 or more times, it's
        impossible to place all its occurrences into nums1 and nums2 while keeping
        the elements in both subarrays distinct.

        Therefore, the condition for a successful split is that the frequency of every
        number in `nums` must be less than or equal to 2.
        """

        # 1. Count the frequency of each number in the input array `nums`.
        #    `collections.Counter` is ideal for this, creating a hash map of
        #    element -> count in O(N) time.
        counts = collections.Counter(nums)

        # 2. Check if any number's frequency is greater than 2.
        #    If so, a valid split is impossible. If all numbers have a frequency
        #    of 2 or less, a split is always possible.
        #    The `all()` function provides a concise and readable way to verify this
        #    condition for all elements. It iterates over the counts and returns
        #    true if and only if all of them satisfy the condition `count <= 2`.
        return all(count <= 2 for count in counts.values())