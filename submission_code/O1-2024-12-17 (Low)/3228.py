class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # We have n even, and must remove n/2 from each array,
        # so each array ends up keeping exactly n/2 of its elements.
        # We want the size of the set of all remaining elements (from both arrays) to be as large as possible.
        #
        # Let:
        #   D1 = set of distinct elements in nums1
        #   D2 = set of distinct elements in nums2
        #   c  = |D1 ∩ D2|
        #   unique1 = D1 \ D2  (distinct elements that only appear in nums1)
        #   unique2 = D2 \ D1  (distinct elements that only appear in nums2)
        #
        # We can choose exactly n/2 distinct elements to keep from nums1 and n/2 distinct elements to keep from nums2.
        # We want to maximize the union of those chosen distinct elements.
        #
        # A known greedy approach:
        #  1) First use up to min(len(unique1), n/2) in nums1 (these elements cannot be chosen from nums2 anyway).
        #  2) Similarly use up to min(len(unique2), n/2) in nums2.
        #  3) Let leftover1 = n/2 - chosen_from_unique1, leftover2 = n/2 - chosen_from_unique2.
        #  4) From the common elements, we can choose each common element in at most one array if we want to increase
        #     the overall union count (choosing the same element in both arrays does not help the union size).
        #     So we can pick at most leftover1 + leftover2 distinct common elements across both arrays.
        #  5) The total coverage = (chosen_from_unique1 + chosen_from_unique2) + min(c, leftover1 + leftover2).
        #  6) Finally, we cannot exceed "n" in total because we're keeping n/2 in each array (total n elements).
        #
        # This formula matches all the provided examples.

        n = len(nums1)
        
        set1 = set(nums1)
        set2 = set(nums2)

        # Distinct sets
        common = set1.intersection(set2)
        unique1 = set1 - common
        unique2 = set2 - common

        # Step 1 & 2: pick from unique sets
        pick1 = min(len(unique1), n // 2)  # how many unique elements we keep from nums1
        pick2 = min(len(unique2), n // 2)  # how many unique elements we keep from nums2

        # Step 3: leftover capacity in each array
        leftover1 = (n // 2) - pick1
        leftover2 = (n // 2) - pick2

        # Step 4 & 5: fill from common
        c = len(common)
        coverage = pick1 + pick2 + min(c, leftover1 + leftover2)

        # Step 6: cannot exceed n
        coverage = min(coverage, n)

        return coverage