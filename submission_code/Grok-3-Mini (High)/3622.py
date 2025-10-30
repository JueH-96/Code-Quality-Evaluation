import bisect
from collections import Counter
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if not nums:
            return 0  # Though constraints ensure nums.length >= 1, handling empty case for safety.
        
        sorted_nums = sorted(nums)
        freq_counter = Counter(nums)
        min_num = min(nums)
        max_num = max(nums)
        ans = 0
        
        for t in range(min_num - k, max_num + k + 1):
            low = t - k
            high = t + k
            A_t = bisect.bisect_right(sorted_nums, high) - bisect.bisect_left(sorted_nums, low)
            B_t = freq_counter.get(t, 0)
            current_min = min(A_t, B_t + numOperations)
            if current_min > ans:
                ans = current_min
        
        return ans