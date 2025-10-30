import collections

class Solution:
  def maximumSetSize(self, nums1: list[int], nums2: list[int]) -> int:
    n = len(nums1)
    # k is the number of elements we must keep from each array
    k = n // 2

    set1 = set(nums1)
    set2 = set(nums2)

    len_set1 = len(set1)
    len_set2 = len(set2)
    
    # Calculate counts of elements based on their presence in set1 and set2
    # common_count: number of elements present in both set1 and set2
    # s1_only_count: number of elements present only in set1 (i.e., in set1 but not set2)
    # s2_only_count: number of elements present only in set2 (i.e., in set2 but not set1)
    
    common_elements_set = set1.intersection(set2)
    common_count = len(common_elements_set)

    s1_only_count = len_set1 - common_count
    s2_only_count = len_set2 - common_count
    
    # Determine the number of distinct elements we can aim to pick from nums1's unique elements (set1).
    # We must pick k elements from nums1. The number of distinct ones among them is at most k.
    # Also, it cannot be more than the total distinct elements available in nums1 (len_set1).
    # So, we can effectively choose `distinct_target_from_set1` distinct elements for our conceptual "kept set" from nums1.
    distinct_target_from_set1 = min(len_set1, k)
    
    # Similarly for nums2
    distinct_target_from_set2 = min(len_set2, k)

    # Now, we decide which specific distinct elements to pick to maximize the final union set size.
    # The strategy is to prioritize elements that are unique to one set, as they are less likely to cause overlaps
    # and thus contribute more readily to increasing the final set size.

    # Contribution from elements unique to set1:
    # From the `distinct_target_from_set1` elements we want to pick for set1's contribution,
    # we first pick as many as possible from `s1_only_count`.
    picked_s1_only = min(distinct_target_from_set1, s1_only_count)
    # Any remaining slots in `distinct_target_from_set1` must be filled by common elements.
    s1_needs_common = distinct_target_from_set1 - picked_s1_only
    
    # Contribution from elements unique to set2:
    # Similarly, for set2's contribution.
    picked_s2_only = min(distinct_target_from_set2, s2_only_count)
    # Any remaining slots in `distinct_target_from_set2` must be filled by common elements.
    s2_needs_common = distinct_target_from_set2 - picked_s2_only
    
    # Contribution from common elements:
    # set1's choices need `s1_needs_common` distinct elements from the common pool.
    # set2's choices need `s2_needs_common` distinct elements from the common pool.
    # The total number of distinct common elements that will be part of the final merged set
    # is the number of elements in the union of these choices from the common pool.
    # This is min(available_common_elements, sum_of_needs_from_common_pool).
    picked_common_total = min(common_count, s1_needs_common + s2_needs_common)
    
    # The maximum size of the final set `s` is the sum of distinct elements obtained from these three categories:
    # 1. `picked_s1_only`: Elements chosen from set1 that are unique to set1.
    # 2. `picked_s2_only`: Elements chosen from set2 that are unique to set2.
    # 3. `picked_common_total`: Distinct common elements chosen by either set1's choices or set2's choices (or both).
    max_set_size = picked_s1_only + picked_s2_only + picked_common_total
    
    return max_set_size