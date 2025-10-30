from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        common = set1 & set2
        onlyA = set1 - set2
        onlyB = set2 - set1
        
        T = min(len(onlyA), n // 2) + min(len(onlyB), n // 2)
        ans = min(T + len(common), n)
        return ans