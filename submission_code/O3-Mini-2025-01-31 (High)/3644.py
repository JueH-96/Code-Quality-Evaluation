from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        best = float('inf')
        # Create prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Iterate over all subarrays with lengths between l and r
        for i in range(n):
            for length in range(l, r + 1):
                if i + length <= n:
                    current_sum = prefix[i + length] - prefix[i]
                    if current_sum > 0:
                        best = min(best, current_sum)
        
        return best if best != float('inf') else -1