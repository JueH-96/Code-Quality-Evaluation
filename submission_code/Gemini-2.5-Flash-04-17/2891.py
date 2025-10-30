from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Sort the array. This allows us to use a sliding window
        # to find the longest contiguous subarray where the difference
        # between the maximum and minimum elements is at most 2*k.
        nums.sort()

        # Initialize the left pointer of the sliding window
        left = 0
        # Initialize the maximum beauty found so far.
        # The beauty is the length of the longest subsequence of equal elements.
        # This is equivalent to the maximum number of original elements
        # that can be transformed into the same value.
        max_beauty = 0
        # Get the number of elements in the array
        n = len(nums)

        # Iterate through the array with the right pointer of the sliding window.
        # The right pointer represents the potential maximum element in the current window.
        for right in range(n):
            # The current window is nums[left...right].
            # In the sorted array, nums[left] is the minimum element
            # and nums[right] is the maximum element in this window.

            # The condition for a set of numbers to be transformable into the same value X
            # is that the difference between the maximum and minimum value in the set
            # is at most 2*k.
            # For the window [left, right], this condition is nums[right] - nums[left] <= 2*k.

            # If the current window [left, right] is invalid, it means the difference
            # between the maximum and minimum elements is greater than 2*k.
            # i.e., nums[right] - nums[left] > 2*k.
            # To make the window valid for the current `right` element, we must
            # remove the smallest element `nums[left]` by moving the left pointer to the right.
            # We continue shrinking the window from the left as long as it remains invalid.
            while nums[right] - nums[left] > 2 * k:
                left += 1

            # After the while loop, the window [left, right] is guaranteed to be valid.
            # It represents the longest contiguous subarray ending at `right`
            # where the difference between the max and min is at most 2*k.
            # The length of this window is `right - left + 1`.
            # This length is the maximum number of elements we can pick from the
            # original array (corresponding to the values in nums[left...right] in sorted order)
            # that can all be transformed into the same value.
            # This length is a potential candidate for the maximum beauty.
            max_beauty = max(max_beauty, right - left + 1)

        # After iterating through all possible end points `right`, max_beauty holds
        # the maximum length of a valid window found. This is the maximum number
        # of elements from the original array that can be transformed into the same value,
        # which is the maximum possible beauty.
        return max_beauty