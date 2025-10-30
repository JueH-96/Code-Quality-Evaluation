import math # Required for float('inf')
from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_positive_sum = float('inf') # Initialize with a value larger than any possible sum

        n = len(nums)

        # Iterate through all possible starting indices of subarrays
        for i in range(n):
            current_sum = 0
            # Iterate through all possible ending indices of subarrays, starting from 'i'
            for j in range(i, n):
                current_sum += nums[j] # Add the current element to the running sum
                current_length = j - i + 1 # Calculate the length of the current subarray

                # Check if the current subarray's length is within the specified range [l, r]
                if l <= current_length <= r:
                    # Check if the current subarray's sum is positive
                    if current_sum > 0:
                        # If both conditions are met, update min_positive_sum
                        min_positive_sum = min(min_positive_sum, current_sum)
        
        # After checking all possible subarrays, determine the return value
        if min_positive_sum == float('inf'):
            # No subarray met the criteria (positive sum and length in range)
            return -1
        else:
            # Return the smallest positive sum found.
            # Cast to int because min_positive_sum might be a float (e.g., 1.0) due to float('inf') initialization.
            return int(min_positive_sum)