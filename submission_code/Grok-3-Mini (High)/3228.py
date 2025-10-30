class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set_nums1 = set(nums1)
        set_nums2 = set(nums2)
        unique_nums1 = len(set_nums1)
        unique_nums2 = len(set_nums2)
        unique_union = len(set_nums1 | set_nums2)
        max_size_nums1 = min(n // 2, unique_nums1)
        max_size_nums2 = min(n // 2, unique_nums2)
        return min(max_size_nums1 + max_size_nums2, unique_union)