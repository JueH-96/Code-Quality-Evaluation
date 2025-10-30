import math # math is not needed, but was in the original thought process
from typing import List

class Solution:
    """
    Finds the minimum value of a partition for the given array nums.

    The problem asks us to partition the array `nums` into two non-empty subarrays,
    `nums1` and `nums2`, such that every element of `nums` belongs to exactly one
    of the subarrays. We need to minimize the value of the partition, which is
    defined as `|max(nums1) - min(nums2)|`.

    Let the sorted version of the array `nums` be `s_1, s_2, ..., s_n`.
    Consider the differences between adjacent elements in the sorted array:
    `d_i = s_{i+1} - s_i` for `i = 0, ..., n-2`.
    Let `min_d = min(d_i)` over all valid `i`.

    We claim that the minimum possible partition value is exactly `min_d`.

    Proof sketch:
    1. Achievability: We can construct a partition that achieves this value.
       Let `k` be an index such that `s_{k+1} - s_k = min_d`.
       Consider the partition `nums1 = {s_1, ..., s_k}` and `nums2 = {s_{k+1}, ..., s_n}`.
       Since `n >= 2` (from constraints), `k` can range from `0` to `n-2`.
       If `k=0`, `nums1={s_1}` and `nums2={s_2, ..., s_n}`. Both non-empty.
       If `k=n-2`, `nums1={s_1, ..., s_{n-1}}` and `nums2={s_n}`. Both non-empty.
       For this partition, `max(nums1) = s_k` and `min(nums2) = s_{k+1}`.
       The value is `|max(nums1) - min(nums2)| = |s_k - s_{k+1}| = s_{k+1} - s_k = min_d`.
       So, the minimum partition value is at most `min_d`.

    2. Lower Bound: We show that any partition has a value of at least `min_d`.
       Let (`A`, `B`) be any valid partition of `nums`. Let `a = max(A)` and `b = min(B)`.
       The value of this partition is `v = |a - b|`.
       Since `A` and `B` form a partition of the sorted elements `s_1, ..., s_n`,
       there must exist an index `i` (where `0 <= i <= n-2`) such that `s_i` and
       `s_{i+1}` belong to different sets.
       Case 1: `s_i` is in `A` and `s_{i+1}` is in `B`.
           Then `a = max(A) >= s_i`.
           And `b = min(B)`. Since `s_{i+1}` is in `B`, `b <= s_{i+1}`.
           If `a >= b`, then `v = a - b`. We cannot directly relate this to `s_{i+1} - s_i`.
           If `a < b`, then `v = b - a`. We know `b <= s_{i+1}` and `a >= s_i`.
           So `v = b - a <= s_{i+1} - s_i = d_i`. This doesn't prove `v >= min_d`.
       Case 2: `s_i` is in `B` and `s_{i+1}` is in `A`.
           Then `a = max(A) >= s_{i+1}`.
           And `b = min(B)`. Since `s_i` is in `B`, `b <= s_i`.
           In this case, `b <= s_i < s_{i+1} <= a`. This implies `a > b`.
           The value `v = a - b`.
           We have `a >= s_{i+1}` and `b <= s_i`.
           Therefore, `v = a - b >= s_{i+1} - s_i = d_i`.
           Since `d_i` is one of the adjacent differences, `d_i >= min_d`.
           Thus, `v >= min_d`.

       If Case 2 never happens for any `i`, it means that if `s_i` is in `B`, then `s_{i+1}` must also be in `B`.
       This implies that `B` must consist of a suffix of the sorted array, i.e.,
       `B = {s_k, s_{k+1}, ..., s_n}` for some `k`.
       Correspondingly, `A = {s_1, ..., s_{k-1}}`.
       This is the type of partition considered in part 1.
       The value is `v = |max(A) - min(B)| = |s_{k-1} - s_k| = s_k - s_{k-1} = d_{k-1}`.
       Since `d_{k-1} >= min_d`, we have `v >= min_d`.

       In all possible partitions, the value `v` is always greater than or equal to `min_d`.
       Combining with part 1, the minimum value is exactly `min_d`.

    Algorithm:
    1. Sort the input array `nums`.
    2. Iterate through the sorted array and find the minimum difference between adjacent elements.
    3. Return this minimum difference.
    """
    def findValueOfPartition(self, nums: List[int]) -> int:
        """
        Calculates the minimum partition value for the given array nums.

        Args:
            nums: A list of positive integers. The length is guaranteed to be at least 2.

        Returns:
            The integer denoting the minimum value of such partition.
        """
        
        # Sort the array in ascending order.
        # Time complexity: O(n log n), where n is the length of nums.
        # Space complexity: O(n) or O(log n) depending on the sort implementation (Python's Timsort is O(n)).
        nums.sort()

        # Get the number of elements in the array.
        n = len(nums)

        # Initialize the minimum difference. Since n >= 2, we can safely access
        # nums[0] and nums[1]. We initialize min_diff with the first adjacent difference.
        # The difference will always be non-negative since the array is sorted.
        min_diff = nums[1] - nums[0]

        # Iterate through the rest of the adjacent pairs in the sorted array.
        # The loop starts from the second element (index 1) up to the second-to-last element (index n-2).
        # We compare the difference between nums[i+1] and nums[i].
        # Time complexity: O(n).
        # Space complexity: O(1).
        for i in range(1, n - 1):
            # Calculate the difference between the current adjacent elements.
            diff = nums[i+1] - nums[i]

            # If the current difference is smaller than the minimum difference found so far,
            # update the minimum difference.
            # Using min() function is also a concise way: min_diff = min(min_diff, diff)
            if diff < min_diff:
                min_diff = diff

            # Optimization: If the minimum difference found is 0, it means there are duplicate
            # elements adjacent in the sorted array. Since the difference cannot be negative,
            # 0 is the smallest possible value. We can stop the search early.
            if min_diff == 0:
                break

        # The minimum difference found between any adjacent pair in the sorted array
        # is the minimum possible partition value.
        return min_diff