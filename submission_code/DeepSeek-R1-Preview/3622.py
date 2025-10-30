import bisect
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        max_freq = 0
        
        for right in range(n):
            target = nums[right]
            left = bisect.bisect_left(nums, target - k)
            window_size = right - left + 1
            # Find the count of 'target' in [left, right]
            count_left = bisect.bisect_left(nums, target, left, right + 1)
            count_right = bisect.bisect_right(nums, target, left, right + 1)
            count = count_right - count_left
            required_operations = window_size - count
            if required_operations <= numOperations:
                if window_size > max_freq:
                    max_freq = window_size
        
        return max_freq