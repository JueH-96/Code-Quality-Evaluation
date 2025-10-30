from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')

        # Initialize the sliding window
        current_or = 0
        left = 0

        for right in range(n):
            # Add the current element to the bitwise OR
            current_or |= nums[right]

            # Try to minimize the window size
            while (current_or | nums[left]) == current_or:
                current_or ^= nums[left]
                left += 1

            # Update the minimum difference
            min_diff = min(min_diff, abs(k - current_or))

        return min_diff