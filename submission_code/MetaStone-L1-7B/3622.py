import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        max_freq = 0
        max_C = 0
        n = len(nums)
        for i in range(n):
            x = nums[i]
            low = x - k
            high = x + k
            left = bisect.bisect_left(nums, low)
            right = bisect.bisect_right(nums, high)
            C_i = right - left
            initial_count_x = bisect.bisect_right(nums, x) - bisect.bisect_left(nums, x)
            possible = min(C_i, initial_count_x + numOperations)
            if possible > max_freq:
                max_freq = possible
            if C_i > max_C:
                max_C = C_i
        possible_max_C = min(max_C, numOperations)
        return max(max_freq, possible_max_C)