class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        cap1 = cap2 = n // 2
        set1 = set(nums1)
        set2 = set(nums2)
        set_common = set1 & set2
        set_only1 = set1 - set2
        set_only2 = set2 - set1

        # Number of unique numbers we can include from set_only1
        len_only1 = min(len(set_only1), cap1)
        set_only1_included = len_only1
        cap1_remaining = cap1 - len_only1

        # Number of unique numbers we can include from set_only2
        len_only2 = min(len(set_only2), cap2)
        set_only2_included = len_only2
        cap2_remaining = cap2 - len_only2

        # Total capacity remaining
        total_cap_remaining = cap1_remaining + cap2_remaining

        # Number of common unique numbers we can include
        len_common = min(len(set_common), total_cap_remaining)
        common_included = len_common

        # Maximum possible size of the set s
        max_set_size = set_only1_included + set_only2_included + common_included

        return max_set_size