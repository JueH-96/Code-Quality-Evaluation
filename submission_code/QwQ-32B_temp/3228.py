class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = set(nums1)
        s2 = set(nums2)
        common = s1 & s2
        u1 = len(s1 - common)
        u2 = len(s2 - common)
        c = len(common)
        n = len(nums1)
        a_max = min(u1, n // 2)
        b_max = min(u2, n // 2)
        total = a_max + b_max + c
        return min(total, n)