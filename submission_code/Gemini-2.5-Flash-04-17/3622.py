from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        """
        Calculates the maximum possible frequency of any element in nums after performing operations.

        Args:
            nums: An integer array.
            k: An integer, the range of values that can be added [-k, k].
            numOperations: The number of operations to perform.

        Returns:
            The maximum possible frequency.
        """
        # Sort the array to use a sliding window approach.
        # Sorting allows us to consider contiguous subarrays that could potentially
        # be made equal to the largest element in the subarray with minimum total cost.
        nums.sort()
        n = len(nums)
        left = 0
        window_sum = 0
        max_freq = 0

        # Iterate through the array with the right pointer, expanding the window.
        for right in range(n):
            # Add the current element to the window sum.
            window_sum += nums[right]
            
            # The current window is nums[left...right]. We consider the possibility
            # of making all elements in this window equal to the largest element, nums[right].
            # For each element nums[i] in the window (left <= i <= right), the required change
            # to make it equal to nums[right] is nums[right] - nums[i].
            # Since the array is sorted, nums[i] <= nums[right], so the required change is >= 0.
            # The total sum of these required non-negative changes across all elements
            # in the window is:
            # sum(nums[right] - nums[i] for i in range(left, right+1))
            # = (right - left + 1) * nums[right] - sum(nums[left...right])
            current_window_size = right - left + 1
            cost = current_window_size * nums[right] - window_sum

            # We have 'numOperations' operations available. Each operation on a chosen index i
            # allows adding a value d_i in [-k, k].
            # To achieve the total positive cost calculated above, the sum of positive additions
            # across the 'numOperations' chosen indices must be at least 'cost'.
            # The maximum possible total sum of positive additions across numOperations operations
            # is obtained by using numOperations operations, each adding +k, summing up to numOperations * k.
            # So, the window [left, right] can potentially be made equal to nums[right]
            # if the total required positive cost <= numOperations * k.

            # If the cost for the current window exceeds the total available budget for positive changes,
            # we need to shrink the window from the left.
            # The total budget for positive changes from 'numOperations' operations is numOperations * k.
            # This is the crucial constraint: the total increase needed must be covered by the total
            # positive capacity from numOperations operations.
            while cost > numOperations * k:
                # Remove the leftmost element from the window sum.
                window_sum -= nums[left]
                # Shrink the window by moving the left pointer.
                left += 1
                # Recalculate the cost for the new, smaller window [left, right].
                # The target remains nums[right].
                current_window_size = right - left + 1
                cost = current_window_size * nums[right] - window_sum


            # If the loop finishes, it means the current window [left, right] can be made
            # equal to nums[right] within the given number of operations and k constraint.
            # The frequency achieved is the size of the current window.
            max_freq = max(max_freq, current_window_size)

        # After iterating through all possible right endpoints, max_freq holds the
        # maximum achievable frequency.
        return max_freq