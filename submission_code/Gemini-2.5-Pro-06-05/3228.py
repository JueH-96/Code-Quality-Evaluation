from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the maximum possible size of a set formed by removing
        n/2 elements from both nums1 and nums2.

        The strategy is to categorize unique numbers into three disjoint sets:
        1. Numbers unique to nums1 (s1_only)
        2. Numbers unique to nums2 (s2_only)
        3. Numbers common to both (common)

        We greedily decide how many elements to keep from each category to maximize
        the final set size, subject to the constraint of keeping n/2 elements
        from each original array.
        """
        n = len(nums1)
        n_half = n // 2

        s1 = set(nums1)
        s2 = set(nums2)

        # Count of elements unique to nums1, unique to nums2, and common to both.
        common_count = len(s1.intersection(s2))
        s1_only_count = len(s1) - common_count
        s2_only_count = len(s2) - common_count
        
        # To maximize the final set size, we prioritize keeping elements that are
        # unique to each array.
        
        # From nums1, we can keep at most n/2 elements. We should use these slots
        # to keep as many elements from s1_only as possible.
        can_keep_from_s1_only = min(s1_only_count, n_half)
        
        # Similarly, for nums2, we prioritize keeping elements from s2_only.
        can_keep_from_s2_only = min(s2_only_count, n_half)
        
        # The numbers from s1_only and s2_only are disjoint. After taking these,
        # we can potentially add all common numbers to form our final set.
        potential_size = can_keep_from_s1_only + can_keep_from_s2_only + common_count
        
        # However, the total number of elements we keep from both arrays is n.
        # The size of the resulting set cannot exceed n.
        # Therefore, the final answer is the minimum of the potential size and n.
        # This logic works because we can always fill the remaining slots with
        # common elements (or duplicates if common elements run out) to satisfy
        # the condition of keeping n/2 elements from each array.
        return min(potential_size, n)