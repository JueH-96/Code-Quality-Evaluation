class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1) // 2
        set1 = set(nums1)
        set2 = set(nums2)

        # Find common elements
        common = set1 & set2

        # Find unique elements in each set
        unique1 = set1 - common
        unique2 = set2 - common

        # Calculate the maximum number of unique elements we can keep from each set
        max_unique1 = min(len(unique1), n)
        max_unique2 = min(len(unique2), n)

        # Calculate the number of common elements we can keep
        max_common = min(len(common), n - max_unique1 + n - max_unique2)

        # The total size of the set will be the sum of unique and common elements we can keep
        return max_unique1 + max_unique2 + max_common