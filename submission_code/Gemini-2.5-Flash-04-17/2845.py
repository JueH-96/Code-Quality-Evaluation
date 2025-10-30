from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order.
        # Let the sorted array be s_1, s_2, ..., s_n.
        # The problem asks to partition nums into two non-empty arrays, nums1 and nums2,
        # to minimize |max(nums1) - min(nums2)|.

        # Consider any partition (nums1, nums2). Since both are non-empty,
        # there must be at least one element from nums in nums1 and at least one in nums2.
        # Consider the sorted array s. There must exist an index k (0 <= k <= n-2)
        # such that s[k] belongs to one set (say nums1) and s[k+1] belongs to the other set (say nums2).
        # If no such index exists, then all elements s[0], ..., s[n-1] must belong to the same set,
        # which would make the other set empty, violating the non-empty constraint.

        # So, for any valid partition, there exists an index k such that s[k] is in nums1
        # and s[k+1] is in nums2 (or vice versa).
        # Let M = max(nums1) and m = min(nums2).
        # If s[k] is in nums1, then M >= s[k].
        # If s[k+1] is in nums2, then m <= s[k+1].

        # Consider the difference between M and m, |M - m|.
        # We have M >= s[k] and m <= s[k+1].
        # Let's look at the difference s[k+1] - s[k]. Since the array is sorted, s[k+1] - s[k] >= 0.
        # We can write s[k+1] - s[k] = (s[k+1] - m) + (m - M) + (M - s[k]).
        # Since s[k+1] >= m, s[k+1] - m >= 0.
        # Since M >= s[k], M - s[k] >= 0.
        # Thus, s[k+1] - s[k] = (non-negative) + (m - M) + (non-negative) >= (m - M).
        # This inequality can be rewritten as M - m >= -(s[k+1] - s[k]) = s[k] - s[k+1].
        # Taking the absolute value, |M - m| >= |s[k] - s[k+1]| = s[k+1] - s[k] (since s[k+1] >= s[k]).

        # This shows that for any partition, the value |max(nums1) - min(nums2)| is greater than
        # or equal to s[k+1] - s[k] for some index k.
        # Therefore, |max(nums1) - min(nums2)| >= min_{0 <= k <= n-2} (s[k+1] - s[k]).

        # Now we need to show that this minimum difference is achievable.
        # Consider the partition where nums1 contains the first k elements of the sorted array
        # and nums2 contains the remaining elements, i.e., nums1 = {s[0], ..., s[k]}
        # and nums2 = {s[k+1], ..., s[n-1]}, for some 0 <= k <= n-2.
        # For this partition, max(nums1) = s[k] and min(nums2) = s[k+1].
        # The value of this partition is |s[k] - s[k+1]| = s[k+1] - s[k].
        # By choosing the index k that minimizes s[k+1] - s[k], we achieve the minimum possible value.

        # Thus, the minimum value of the partition is exactly the minimum difference
        # between adjacent elements in the sorted array.

        nums.sort() # Sort the array

        n = len(nums)
        # Initialize minimum difference with the difference of the first two elements.
        # Constraints guarantee n >= 2, so nums[0] and nums[1] exist.
        min_diff = nums[1] - nums[0]

        # Iterate through the sorted array from the second pair onwards
        # to find the minimum difference between adjacent elements.
        for i in range(1, n - 1):
            diff = nums[i+1] - nums[i]
            # Since the array is sorted, diff >= 0, no need for absolute value.
            if diff < min_diff:
                min_diff = diff

        return min_diff