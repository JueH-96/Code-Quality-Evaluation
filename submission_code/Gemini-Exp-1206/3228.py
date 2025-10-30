class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        s1 = set(nums1)
        s2 = set(nums2)
        common = len(s1.intersection(s2))
        unique1 = len(s1) - common
        unique2 = len(s2) - common
        remove1 = max(0, n // 2 - unique1)
        remove2 = max(0, n // 2 - unique2)
        common_used = min(common, n - remove1 - remove2)
        return min(n, unique1 + unique2 + common_used)