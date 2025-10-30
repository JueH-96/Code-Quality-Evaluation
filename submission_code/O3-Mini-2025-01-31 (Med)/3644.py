from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Create prefix sum array for efficient subarray sum computation
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        min_sum = float('inf')
        # Explore all subarrays with length between l and r
        for i in range(n):
            for length in range(l, r + 1):
                if i + length <= n:
                    curr_sum = prefix[i + length] - prefix[i]
                    if curr_sum > 0:
                        min_sum = min(min_sum, curr_sum)
        
        return min_sum if min_sum != float('inf') else -1