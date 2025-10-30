import collections
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        """
        Calculates the maximum sum among all almost unique subarrays of length k in the given list nums.
        An almost unique subarray has at least m distinct elements.

        Args:
            nums: A list of integers.
            m: The minimum number of distinct elements required for a subarray to be almost unique.
            k: The length of the subarrays to consider.

        Returns:
            The maximum sum found among almost unique subarrays of length k.
            Returns 0 if no such subarray exists.
        
        Constraints guarantee:
            1 <= nums.length <= 2 * 10^4
            1 <= m <= k <= nums.length
            1 <= nums[i] <= 10^9
        """
        n = len(nums)
        
        # Initialize the maximum sum found so far to 0.
        # If no almost unique subarray is found, this default value will be returned.
        max_found_sum = 0
        # Initialize the sum of the current sliding window.
        current_window_sum = 0
        
        # Use collections.Counter to efficiently track the frequency of elements within the current window.
        # The keys are the numbers in the window, and the values are their counts.
        window_counts = collections.Counter()

        # --- Step 1: Process the first window (indices 0 to k-1) ---
        # Calculate the initial sum and element counts for the first subarray of length k.
        # The loop iterates k times, covering indices 0, 1, ..., k-1.
        for i in range(k):
            num = nums[i]
            current_window_sum += num
            window_counts[num] += 1

        # --- Step 2: Check the first window ---
        # Check if the initial window satisfies the condition (>= m distinct elements).
        # len(window_counts) gives the number of distinct elements currently in the counter (i.e., keys with count > 0).
        if len(window_counts) >= m:
            # If the condition is met, this window's sum is the first candidate for the maximum sum.
            max_found_sum = current_window_sum

        # --- Step 3: Slide the window ---
        # Slide the window one element at a time, from index k up to n-1.
        # The loop variable 'i' represents the index of the element *entering* the window (the rightmost element).
        # In each iteration, the window slides from nums[i-k : i] to nums[i-k+1 : i+1].
        for i in range(k, n):
            # Identify the element leaving the window (at index i - k). This was the leftmost element of the previous window.
            left_num = nums[i - k]
            # Identify the element entering the window (at index i). This is the new rightmost element.
            right_num = nums[i]

            # Update the window sum efficiently: subtract the value of the element leaving, add the value of the element entering.
            current_window_sum -= left_num
            current_window_sum += right_num

            # Update the frequency counts in the window_counts counter.
            # Decrease the count for the element leaving the window.
            window_counts[left_num] -= 1
            # If the count of the element that just left becomes zero, it means this element is no longer present in the window.
            # Remove its key from the counter to ensure len(window_counts) accurately reflects the number of distinct elements *currently* in the window.
            # Using 'del' is important here; simply having a zero count would keep the key in the counter.
            if window_counts[left_num] == 0:
                del window_counts[left_num]

            # Increase the count for the element entering the window.
            # If the element was already present, its count increments. If it was not present, it's added with a count of 1.
            window_counts[right_num] += 1

            # --- Step 4: Check the current window ---
            # Check if the current window (after sliding and updates) satisfies the almost unique condition (>= m distinct elements).
            if len(window_counts) >= m:
                # If the condition is met, compare its sum (current_window_sum) with the maximum sum found so far (max_found_sum).
                # Update max_found_sum if the current window's sum is greater.
                max_found_sum = max(max_found_sum, current_window_sum)

        # --- Step 5: Return the result ---
        # After iterating through all possible windows of length k, return the overall maximum sum found.
        # If no window met the condition, max_found_sum remains 0, which is the correct output in that case.
        return max_found_sum