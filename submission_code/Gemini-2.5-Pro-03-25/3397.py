import math
from typing import List

class Solution:
    """
    This class provides a solution to find the integer 'x' added to each element
    of nums1 to make it equal to nums2.
    """
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the integer x such that adding x to each element of nums1
        results in an array equal to nums2 (meaning they have the same elements
        with the same frequencies).

        The problem states that `nums2` is formed by adding an integer `x` to
        each element of `nums1`. Let `n` be the length of the arrays.
        If we sum the elements of both arrays:
        sum(nums2) = sum(nums1[i] + x for i in range(n))
        sum(nums2) = sum(nums1) + sum(x for i in range(n))
        sum(nums2) = sum(nums1) + n * x

        From this, we can derive x:
        n * x = sum(nums2) - sum(nums1)
        x = (sum(nums2) - sum(nums1)) / n

        Since the problem guarantees that such an integer x exists, the division
        (sum(nums2) - sum(nums1)) / n will result in an integer. We use integer
        division `//` to calculate x.

        Args:
            nums1: The first list of integers.
            nums2: The second list of integers, which is derived from nums1 by
                   adding x to each element. nums1 and nums2 have equal length.

        Returns:
            The integer x.

        Constraints:
            1 <= nums1.length == nums2.length <= 100
            0 <= nums1[i], nums2[i] <= 1000
            An integer x satisfying the condition is guaranteed to exist.

        Complexity:
            Time: O(n), where n is the length of the arrays. This is dominated by
                  the time taken to calculate the sums of the two arrays.
            Space: O(1), as we only use a few variables to store the sums,
                   the length, and the result. No additional data structures
                   proportional to the input size are used.
        """

        # Calculate the sum of elements in nums1.
        # The sum() function iterates through the list once, taking O(n) time.
        sum1 = sum(nums1)

        # Calculate the sum of elements in nums2.
        # This also takes O(n) time.
        sum2 = sum(nums2)

        # Get the length of the arrays. They are guaranteed to have equal length.
        # len() on a list takes O(1) time.
        n = len(nums1)

        # Calculate the total difference between the sums of the two arrays.
        # This difference is equal to n * x.
        diff_sum = sum2 - sum1

        # Calculate x by dividing the total difference by the number of elements.
        # The problem guarantees x is an integer, so this division is exact.
        # We use integer division // to ensure the result is an integer type.
        # The constraint n >= 1 ensures we are not dividing by zero.
        x = diff_sum // n

        # Return the calculated integer x.
        return x