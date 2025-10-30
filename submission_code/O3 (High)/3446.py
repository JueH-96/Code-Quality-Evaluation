from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        """
        Counts the number of pairs (i, j) such that nums1[i] is divisible by
        nums2[j] * k.
        """
        total = 0
        for a in nums1:
            for b in nums2:
                if a % (b * k) == 0:
                    total += 1
        return total