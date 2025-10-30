from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the maximum possible size of the set formed by the remaining elements
        after removing n/2 elements from nums1 and n/2 elements from nums2.

        Args:
            nums1: A 0-indexed integer array of even length n.
            nums2: A 0-indexed integer array of even length n.

        Returns:
            The maximum possible size of the resulting set.
        """
        n = len(nums1)
        n_half = n // 2

        # Convert lists to sets to easily get unique elements and perform set operations
        set1 = set(nums1)
        set2 = set(nums2)

        # Get unique elements present only in nums1
        only1_set = set1 - set2
        # Get unique elements present only in nums2
        only2_set = set2 - set1
        # Get unique elements present in both nums1 and nums2
        intersect_set = set1 & set2

        # To maximize the size of the union set, we should prioritize unique elements
        # that are only present in one of the arrays, as they can only be obtained
        # from that specific array.

        # Number of unique elements from only1_set we can possibly include.
        # This is limited by the total number of unique elements in only1_set
        # and the number of elements we are allowed to keep from nums1 (n_half).
        count_only1 = min(len(only1_set), n_half)

        # Same logic for unique elements only in nums2.
        count_only2 = min(len(only2_set), n_half)

        # After deciding to include 'count_only1' distinct elements from only1_set
        # using 'count_only1' slots from nums1, we have n_half - count_only1 slots
        # remaining in nums1. Similarly, we have n_half - count_only2 slots remaining
        # in nums2.
        # The total number of elements we must still pick is (n_half - count_only1)
        # from nums1 and (n_half - count_only2) from nums2.
        # The unique values these remaining elements can introduce to the set s
        # are those from the intersect_set. This is because unique-only elements
        # from only1_set (if not already included up to the n_half limit) cannot be
        # picked from nums2, and vice versa.
        # The number of distinct unique elements from intersect_set we can possibly
        # add is limited by the total number of unique elements in intersect_set
        # and the total number of remaining slots available across both arrays
        # combined, which is (n_half - count_only1) + (n_half - count_only2).
        # We can pick at most this many elements in total using the remaining slots.
        # We can strategically pick these elements to get as many *distinct* common
        # values as possible.
        remaining_slots_total = (n_half - count_only1) + (n_half - count_only2)
        count_common = min(len(intersect_set), remaining_slots_total)

        # The total maximum size of the set s is the sum of the distinct unique
        # elements we managed to include from the three disjoint categories:
        # unique-to-nums1, unique-to-nums2, and common elements.
        total_size = count_only1 + count_only2 + count_common

        # The total size is bounded by the total number of elements kept (n)
        # and the total number of initially unique elements in nums1 U nums2.
        # Our calculation strategy implicitly respects these bounds.
        # We know count_only1 <= n_half, count_only2 <= n_half, and
        # count_common <= remaining_slots_total = n - count_only1 - count_only2.
        # Summing these gives total_size <= count_only1 + count_only2 + (n - count_only1 - count_only2) = n.
        # The total size is also bounded by |set1 U set2|.
        # total_size = min(|only1|, n/2) + min(|only2|, n/2) + min(|intersect|, n - min(|only1|, n/2) - min(|only2|, n/2))
        # Let k1 = min(|only1|, n/2), k2 = min(|only2|, n/2).
        # total_size = k1 + k2 + min(|intersect|, n - k1 - k2).
        # Since k1 <= |only1|, k2 <= |only2|, and min(|intersect|, ...) <= |intersect|,
        # total_size <= |only1| + |only2| + |intersect| = |set1 U set2|.
        # The derived formula gives the maximum achievable size under the constraints.

        return total_size