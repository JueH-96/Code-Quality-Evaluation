from typing import List
from collections import Counter

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if numOperations == 0:
            # If no operations are allowed, return the frequency of the most common element
            return max(Counter(nums).values())

        # Sort the array to facilitate sliding window approach
        nums.sort()

        # Initialize variables
        n = len(nums)
        left = 0
        total_sum = 0
        max_freq = 0

        for right in range(n):
            total_sum += nums[right]

            # While the sum of the current window exceeds the allowed operations
            while (right - left + 1) * nums[right] - total_sum > k * numOperations:
                total_sum -= nums[left]
                left += 1

            # Calculate the maximum possible frequency
            max_freq = max(max_freq, right - left + 1)

        return max_freq