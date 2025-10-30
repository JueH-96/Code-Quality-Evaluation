from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # The problem asks for the maximum sum of a unique subarray
        # from the original array, after potentially deleting elements.
        # Based on the examples, the most plausible interpretation is to find
        # the maximum sum unique contiguous subarray within the *original* array.
        # The ability to delete elements implies that any such unique contiguous
        # subarray from the original array is achievable as a unique contiguous
        # subarray of *some* sequence obtained by deletions (e.g., by deleting
        # all elements outside the chosen subarray that are duplicates of
        # elements inside it, and potentially other negative elements outside
        # the subarray).

        # We can solve the Maximum Sum Unique Subarray problem in the original
        # array using a sliding window and a frequency map.

        n = len(nums)
        # Constraint is 1 <= nums.length, so n >= 1. No need to handle n == 0.

        # Initialize max_sum. A single element is a valid unique subarray.
        # The maximum sum is at least the maximum value in the array.
        # This handles cases where all possible unique subarrays have negative sums
        # by ensuring we consider each individual element as a potential answer.
        max_sum = max(nums)

        current_sum = 0
        freq = {} # Frequency map to store counts of elements in the current window [l, r]
        l = 0     # Left pointer of the sliding window

        # Right pointer iterates through the array
        for r in range(n):
            num = nums[r]

            # Add the current element to the window [l, r]
            freq[num] = freq.get(num, 0) + 1
            current_sum += num

            # If the current element `num` is already present in the window (frequency > 1),
            # shrink the window from the left (increase l) until the duplicate is removed.
            while freq[num] > 1:
                left_num = nums[l]
                freq[left_num] -= 1
                # If the frequency becomes 0, we could optionally remove the key,
                # but it's not strictly necessary as we only check > 1.
                # if freq[left_num] == 0:
                #     del freq[left_num]

                current_sum -= left_num # Remove the element from the sum
                l += 1                  # Move the left pointer

            # At this point, the window `nums[l...r]` contains only unique elements.
            # Update the maximum sum found so far.
            # We take max(max_sum, current_sum). The initialization max_sum = max(nums)
            # handles the base case where the max unique subarray is a single element,
            # including cases where the optimal unique subarray has sum 0 or is negative.
            max_sum = max(max_sum, current_sum)

        return max_sum