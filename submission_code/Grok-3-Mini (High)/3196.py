import math
from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        # Compute prefix sum
        prefix = [0]
        cum_sum = 0
        for num in sorted_nums:
            cum_sum += num
            prefix.append(cum_sum)
        
        # Define check function
        def check(length):
            mid_idx = length // 2
            D = 2 * mid_idx + 1 - length
            for L in range(n - length + 1):
                sum_A = prefix[L + mid_idx + 1] - prefix[L]
                sum_B = prefix[L + length] - prefix[L + mid_idx]
                C_val = sorted_nums[L + mid_idx]
                cost = -sum_A + sum_B + D * C_val
                if cost <= k:
                    return True
            return False
        
        # Binary search for the maximum length
        left = 1
        right = n
        while left <= right:
            mid_len = (left + right) // 2
            if check(mid_len):
                left = mid_len + 1
            else:
                right = mid_len - 1
        
        # After the loop, right is the highest length where check is true
        return right