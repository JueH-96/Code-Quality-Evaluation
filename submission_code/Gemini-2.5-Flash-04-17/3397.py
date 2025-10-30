from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the sum of elements in both arrays.
        # If each element nums1[i] is transformed to nums1[i] + x,
        # then the sum of the new array is sum(nums1) + n * x,
        # where n is the length of the array.
        sum1 = sum(nums1)
        sum2 = sum(nums2)

        # The length of the arrays (they are guaranteed to be equal).
        n = len(nums1)

        # Since adding x to each element of nums1 makes it equal to nums2
        # (in terms of elements and frequencies), the sum of the transformed
        # nums1 must be equal to the sum of nums2.
        # So, sum(nums1) + n * x = sum(nums2).
        # We can solve for x:
        # n * x = sum(nums2) - sum(nums1)
        # x = (sum(nums2) - sum(nums1)) / n

        # The problem guarantees that such an integer x exists.
        # Therefore, the difference (sum2 - sum1) must be perfectly divisible by n.
        # We use integer division // to ensure the result is an integer.
        x = (sum2 - sum1) // n

        return x