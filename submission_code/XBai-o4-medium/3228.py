class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        set1 = set(nums1)
        set2 = set(nums2)
        n = len(nums1)
        max_from_1 = min(len(set1), n // 2)
        max_from_2 = min(len(set2), n // 2)
        unique_total = len(set1.union(set2))
        return min(unique_total, max_from_1 + max_from_2)