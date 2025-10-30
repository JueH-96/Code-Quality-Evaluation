from typing import List
import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        first_occurrence = [0] * n
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                first_occurrence[i] = first_occurrence[i-1]
            else:
                first_occurrence[i] = i
        max_freq = 1
        for j in range(n):
            target = nums[j] - k
            left_i = bisect.bisect_left(nums, target)
            window_size = j - left_i + 1
            start = max(first_occurrence[j], left_i)
            required_ops = start - left_i
            if required_ops <= numOperations:
                if window_size > max_freq:
                    max_freq = window_size
        return max_freq