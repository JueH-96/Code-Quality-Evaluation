from typing import List
import bisect

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        case2_max = self.compute_case2_max(nums, k, numOperations)
        case1_max = self.compute_case1_max(nums, k, numOperations)
        return max(case1_max, case2_max)
    
    def compute_case2_max(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_window_size = 0
        left = 0
        for right in range(n):
            while nums[right] - nums[left] > 2 * k:
                left += 1
            current_window_size = right - left + 1
            if current_window_size > max_window_size:
                max_window_size = current_window_size
        return min(max_window_size, numOperations)
    
    def compute_case1_max(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        case1_max = 0
        prev = None
        for i in range(n):
            x = nums[i]
            if x == prev:
                continue
            prev = x
            low = x - k
            high = x + k
            left = bisect.bisect_left(nums, low)
            right_b = bisect.bisect_right(nums, high)
            count_x = right_b - left
            orig_left = bisect.bisect_left(nums, x)
            orig_right = bisect.bisect_right(nums, x)
            original_x = orig_right - orig_left
            possible_frequency = original_x + min(count_x - original_x, numOperations)
            if possible_frequency > case1_max:
                case1_max = possible_frequency
        return case1_max