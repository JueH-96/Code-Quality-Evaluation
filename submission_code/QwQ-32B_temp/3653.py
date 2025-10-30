from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_prefix = [float('inf')] * k
        min_prefix[0] = prefix[0]  # Initial value for remainder 0
        
        max_sum = -float('inf')
        
        for m in range(1, n + 1):
            r = m % k
            # Check if there is a previous prefix to form a valid subarray
            if min_prefix[r] != float('inf'):
                current = prefix[m] - min_prefix[r]
                if current > max_sum:
                    max_sum = current
            # Update the minimum prefix for the current remainder
            if prefix[m] < min_prefix[r]:
                min_prefix[r] = prefix[m]
        
        return max_sum