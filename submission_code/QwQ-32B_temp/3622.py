import bisect
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        max_freq = 0
        n = len(nums)
        for T in nums:
            # Find the left boundary of the range [T - k, T + k]
            left = bisect.bisect_left(nums, T - k)
            # Find the right boundary of the range [T - k, T + k]
            right_val = T + k
            right = bisect.bisect_right(nums, right_val) - 1
            if left > right:
                count_in_range = 0
            else:
                count_in_range = right - left + 1
            # Calculate the original count of T
            original_count = bisect.bisect_right(nums, T) - bisect.bisect_left(nums, T)
            # Compute the possible maximum frequency for this T
            possible = min(count_in_range, original_count + numOperations)
            if possible > max_freq:
                max_freq = possible
        return max_freq