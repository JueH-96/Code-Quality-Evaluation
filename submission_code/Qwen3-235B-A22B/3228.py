class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        set1 = set(nums1)
        set2 = set(nums2)
        common = set1 & set2
        a = len(set1 - common)
        b = len(set2 - common)
        c = len(common)
        n = len(nums1)
        k1 = n // 2
        k2 = n // 2
        
        sum_ab = min(a, k1) + min(b, k2)
        sum_abc = sum_ab + c
        return min(sum_abc, k1 + k2)