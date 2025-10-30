from collections import Counter
from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        n_remove = n // 2

        counts1 = Counter(nums1)
        counts2 = Counter(nums2)

        unique_elements = set(nums1 + nums2)
        max_set_size = 0

        for i in range(1 << n):
            removed_indices1 = []
            remaining_nums1 = []
            for j in range(n):
                if (i >> j) & 1:
                    removed_indices1.append(j)
                else:
                    remaining_nums1.append(nums1[j])

            if len(removed_indices1) != n_remove:
                continue

            for k in range(1 << n):
                removed_indices2 = []
                remaining_nums2 = []
                for l in range(n):
                    if (k >> l) & 1:
                        removed_indices2.append(l)
                    else:
                        remaining_nums2.append(nums2[l])

                if len(removed_indices2) != n_remove:
                    continue

                s = set(remaining_nums1 + remaining_nums2)
                max_set_size = max(max_set_size, len(s))

        return max_set_size