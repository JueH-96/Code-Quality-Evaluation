class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        s1 = set(nums1)
        s2 = set(nums2)
        total_unique = len(s1.union(s2))
        a = min(len(s1), n // 2)
        b = min(len(s2), n // 2)
        return min(a + b, total_unique)