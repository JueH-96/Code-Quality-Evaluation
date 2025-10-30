class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        common = set1.intersection(set2)
        unique1 = len(set1) - len(common)
        unique2 = len(set2) - len(common)
        return min(n, len(common) + min(unique1, n // 2) + min(unique2, n // 2))