from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        k = n // 2

        set1 = set(nums1)
        set2 = set(nums2)

        # Elements only in nums1
        only1 = set1 - set2
        # Elements only in nums2
        only2 = set2 - set1
        # Common elements
        common = set1 & set2

        x = len(only1)
        y = len(only2)
        z = len(common)

        # From each array we can keep at most k items.
        # To maximize distinct union, first take as many "only" elements as possible.
        take1 = min(x, k)
        take2 = min(y, k)

        # Remaining slots in each array can be filled with common elements
        rem1 = k - take1
        rem2 = k - take2

        # Across both arrays we have rem1+rem2 slots for common elements,
        # but there are only z distinct common elements.
        union_common = min(z, rem1 + rem2)

        # Total distinct = distinct from only1 + only2 + those common we managed to include
        return take1 + take2 + union_common