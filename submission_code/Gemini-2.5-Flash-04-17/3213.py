from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Step 1: Find the maximum element in the array
        # This is necessary because the condition is based on the maximum element
        max_num = max(nums)
        n = len(nums)

        # Step 2: Initialize variables for the sliding window approach
        total_count = 0  # Accumulator for the total number of valid subarrays
        left = 0         # Left pointer of the sliding window [left:right+1]
        count = 0        # Counter for the occurrences of the maximum element within the current window

        # Step 3: Iterate through the array using the right pointer
        # The right pointer expands the sliding window one element at a time
        for right in range(n):
            # Step 4: Update the count of max_num when the right pointer advances
            # If the current element at `right` is the maximum element, increment the count
            if nums[right] == max_num:
                count += 1

            # Step 5: While the count of max_num in the current window [left:right+1] is at least k
            # This means the current window satisfies the condition.
            # We need to find the number of valid subarrays that end at the current `right` index.
            # A subarray ending at `right` starts at some index `i` where 0 <= i <= right.
            # The condition is count(max_num, nums[i:right+1]) >= k.
            # As `i` decreases from `right` down to 0, the count of max_num in `nums[i:right+1]` is non-decreasing.
            # The `while` loop shrinks the window from the left (`left++`) as long as the condition `count >= k` holds
            # for the window `nums[left:right+1]`.
            # When the loop terminates, `left` will be the smallest index such that the window
            # `nums[left:right+1]` contains *less than* `k` occurrences of `max_num`.
            # This implies that for any starting index `i` such that `0 <= i < left` (the value of `left`
            # after the while loop finishes), the subarray `nums[i:right+1]` must contain at least `k`
            # occurrences of `max_num`.
            # The number of such starting indices `i` is exactly `left`.
            while count >= k:
                # If the element at the left pointer is the maximum, decrement the count
                # as it is about to be removed from the window (conceptually)
                if nums[left] == max_num:
                    count -= 1
                # Shrink the window from the left by moving the left pointer forward
                left += 1

            # Step 6: Add the number of valid subarrays ending at the current `right`
            # As explained above, the number of valid starting indices `i` (0 <= i <= right)
            # for subarrays ending at `right` is equal to the current value of the `left` pointer
            # after the shrinking `while` loop.
            # The number of valid subarrays ending at `right` is `left`.
            total_count += left

        # Step 7: Return the accumulated total count of valid subarrays
        return total_count