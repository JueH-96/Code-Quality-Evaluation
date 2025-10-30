from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        # Iterate over all possible starting points of subarrays
        for start in range(n):
            current_sum = 0
            # Check subarrays of length from l to r
            for end in range(start, min(start + r, n)):
                current_sum += nums[end]
                if end - start + 1 >= l and current_sum > 0:
                    min_sum = min(min_sum, current_sum)
                    found = True
        
        return min_sum if found else -1