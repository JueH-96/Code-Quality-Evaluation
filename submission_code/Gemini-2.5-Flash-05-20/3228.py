from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        n_half = n // 2

        # Convert arrays to sets to get unique elements
        set1 = set(nums1)
        set2 = set(nums2)

        # Partition the elements into three categories:
        # 1. Elements unique to nums1 (s1_only)
        # 2. Elements unique to nums2 (s2_only)
        # 3. Elements common to both nums1 and nums2 (common)
        s1_only = set1 - set2
        s2_only = set2 - set1
        common = set1 & set2

        # Initialize the count of distinct elements in the final set s
        current_distinct_count = 0

        # Phase 1: Prioritize unique elements from s1_only and s2_only

        # For s1_only elements:
        # We can keep at most n_half elements from nums1.
        # We greedily take unique elements from s1_only first.
        kept_from_s1_only = min(len(s1_only), n_half)
        current_distinct_count += kept_from_s1_only
        
        # Calculate how many slots are left in nums1 to be filled by common elements.
        # These are slots we *must* fill from nums1's side, and they can't be s1_only anymore.
        remaining_slots_for_nums1 = n_half - kept_from_s1_only

        # For s2_only elements:
        # Similarly for nums2 and s2_only elements.
        kept_from_s2_only = min(len(s2_only), n_half)
        current_distinct_count += kept_from_s2_only
        
        # Calculate how many slots are left in nums2 to be filled by common elements.
        remaining_slots_for_nums2 = n_half - kept_from_s2_only

        # Phase 2: Fill remaining slots with common elements

        # We need to fill 'remaining_slots_for_nums1' from nums1 and
        # 'remaining_slots_for_nums2' from nums2, using common elements.
        # To maximize distinctness, we simply take as many *distinct* common elements as possible.
        # The total number of distinct common elements we can possibly add to the set is limited by:
        # 1. The actual number of unique elements in 'common'.
        # 2. The combined capacity for common elements from both arrays.
        kept_from_common = min(len(common), remaining_slots_for_nums1 + remaining_slots_for_nums2)
        current_distinct_count += kept_from_common

        # The result must be at most 'n' because we only keep 'n' elements in total.
        # Our calculation naturally ensures this:
        # kept_from_s1_only <= n_half
        # kept_from_s2_only <= n_half
        # kept_from_common <= remaining_slots_for_nums1 + remaining_slots_for_nums2
        # = (n_half - kept_from_s1_only) + (n_half - kept_from_s2_only)
        # So, current_distinct_count <= kept_from_s1_only + kept_from_s2_only + (n_half - kept_from_s1_only) + (n_half - kept_from_s2_only)
        # = n_half + n_half = n
        # Thus, no explicit 'min(result, n)' is needed, as the result will not exceed 'n'.
        return current_distinct_count