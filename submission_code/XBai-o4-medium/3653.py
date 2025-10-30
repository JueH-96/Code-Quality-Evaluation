from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        min_prev = [float('inf')] * k
        min_prev[0] = 0  # for the initial prefix sum of 0
        current_prefix = 0
        max_sum = float('-inf')
        
        for i in range(len(nums)):
            current_prefix += nums[i]
            r = (i + 1) % k  # remainder for the current prefix index (i+1)
            # Check if there's a valid previous prefix sum for this remainder
            if min_prev[r] != float('inf'):
                candidate = current_prefix - min_prev[r]
                if candidate > max_sum:
                    max_sum = candidate
            # Update the min_prev for the current remainder
            if current_prefix < min_prev[r]:
                min_prev[r] = current_prefix
        
        return max_sum