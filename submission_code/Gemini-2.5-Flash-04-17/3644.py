from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        """
        Finds the minimum sum of a subarray with length between l and r (inclusive)
        and sum greater than 0.

        Args:
            nums: An integer array.
            l: The minimum length of the subarray.
            r: The maximum length of the subarray.

        Returns:
            The minimum positive sum of a valid subarray, or -1 if none exists.
        """
        n = len(nums)
        min_sum = float('inf') # Use infinity to track the minimum positive sum

        # Iterate through all possible start indices
        for i in range(n):
            current_sum = 0
            # Iterate through all possible end indices starting from i
            for j in range(i, n):
                current_sum += nums[j]
                subarray_length = j - i + 1

                # Optimization: If the current subarray's length exceeds the maximum allowed length 'r',
                # then any subsequent subarrays starting at 'i' will also be too long.
                # We can break the inner loop for this starting index 'i'.
                if subarray_length > r:
                    break # Move to the next starting index 'i'

                # Only consider subarrays whose length is at least 'l'
                if subarray_length >= l:
                    # Check if the sum is greater than 0
                    if current_sum > 0:
                        # Update minimum sum found so far if the current sum is smaller
                        min_sum = min(min_sum, current_sum)

        # After checking all relevant subarrays, if min_sum is still infinity,
        # it means no subarray met the criteria (length in [l, r] and sum > 0).
        if min_sum == float('inf'):
            return -1 # Indicate no such subarray exists
        else:
            return int(min_sum) # Return the found minimum positive sum as an integer