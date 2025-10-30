from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_positive = float('inf')
        # Try every starting index
        for start in range(n):
            # Try every allowable subarray length
            for length in range(l, r + 1):
                end = start + length
                if end <= n:
                    s = prefix[end] - prefix[start]
                    if s > 0 and s < min_positive:
                        min_positive = s
        
        return min_positive if min_positive != float('inf') else -1