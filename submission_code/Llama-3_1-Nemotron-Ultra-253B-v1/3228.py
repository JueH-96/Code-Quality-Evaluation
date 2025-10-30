from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        set1 = set(nums1)
        set2 = set(nums2)
        A = set1.intersection(set2)
        B = set1 - set2
        C = set2 - set1
        lenA = len(A)
        lenB = len(B)
        lenC = len(C)
        n = len(nums1)
        sum_max_bc = min(lenB, n // 2) + min(lenC, n // 2)
        return min(sum_max_bc + lenA, n)