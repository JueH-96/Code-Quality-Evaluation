from typing import List
import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_counts = [0] * (n + 1)
        for i in range(n):
            prefix_counts[i + 1] = prefix_counts[i] + 1  # Each element is counted once

        max_freq = 0
        left = 0

        for right in range(n):
            # Expand the window as long as the difference between nums[right] and nums[left] is <= k
            while left <= right and nums[right] - nums[left] > k:
                left += 1
            window_size = right - left + 1
            # Calculate the number of operations needed to make all elements in the window equal to nums[right]
            # Elements already equal to nums[right] don't need operations
            # Elements within [nums[right] - k, nums[right] + k] can be adjusted to nums[right]
            # Since the window already ensures nums[right] - nums[left] <= k,
            # all elements in the window can be adjusted to nums[right] with at most one operation each
            operations_needed = window_size - 1  # To make all elements equal to nums[right], need to adjust window_size - 1 elements
            if operations_needed <= numOperations:
                max_freq = max(max_freq, window_size)
        
        return max_freq