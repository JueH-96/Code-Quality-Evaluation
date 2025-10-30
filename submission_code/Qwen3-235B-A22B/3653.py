from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        current_sum = 0
        max_total = -math.inf
        min_remainder = [math.inf] * k
        min_remainder[0] = 0  # Initial prefix sum of 0 at index 0
        
        for i in range(n):
            current_sum += nums[i]
            j_mod_k = (i + 1) % k  # j is i+1 as prefix_sum[0] is 0
            # Check if there's a valid previous remainder to form a valid subarray
            if min_remainder[j_mod_k] != math.inf:
                candidate = current_sum - min_remainder[j_mod_k]
                if candidate > max_total:
                    max_total = candidate
            # Update the min_remainder for the current remainder if current_sum is smaller
            if current_sum < min_remainder[j_mod_k]:
                min_remainder[j_mod_k] = current_sum
        
        return max_total