import bisect
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        min_num = nums[0]
        max_num = nums[-1]
        max_freq = 0
        min_T = min_num - k
        max_T = max_num + k
        
        for T in range(min_T, max_T + 1):
            # Calculate the number of elements in [T - k, T + k]
            left = bisect.bisect_left(nums, T - k)
            right = bisect.bisect_right(nums, T + k)
            count_in_range = right - left
            
            # Calculate the number of elements equal to T
            left_T = bisect.bisect_left(nums, T)
            right_T = bisect.bisect_right(nums, T)
            count_T = right_T - left_T
            
            possible_ops = count_in_range - count_T
            add = min(numOperations, possible_ops)
            current = count_T + add
            if current > max_freq:
                max_freq = current
        
        return max_freq