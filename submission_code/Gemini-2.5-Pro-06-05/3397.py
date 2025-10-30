from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the integer `x` added to each element of `nums1` to form `nums2`.

        The problem states that `nums2` is formed by adding a constant integer `x`
        to each element of `nums1` and then reordering. This means that if we add
        `x` to the smallest element of `nums1`, we must get the smallest element of `nums2`.

        Let `min1` be the minimum value in `nums1`.
        Let `min2` be the minimum value in `nums2`.

        Then, min1 + x = min2.
        Solving for x, we get: x = min2 - min1.
        
        This logic also applies to the maximum values or the sums of the arrays.
        Using min (or max) is a very efficient O(N) approach.
        """
        
        min_val_nums1 = min(nums1)
        min_val_nums2 = min(nums2)
        
        return min_val_nums2 - min_val_nums1