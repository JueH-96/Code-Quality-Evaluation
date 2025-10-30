class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        common = set1 & set2
        c = len(common)
        a = len(set1)
        b = len(set2)
        o1 = a - c
        o2 = b - c
        n_half = n // 2
        keep_o1 = min(o1, n_half)
        keep_o2 = min(o2, n_half)
        rem1 = n_half - keep_o1
        rem2 = n_half - keep_o2
        keep_common = min(c, rem1 + rem2)
        return keep_o1 + keep_o2 + keep_common