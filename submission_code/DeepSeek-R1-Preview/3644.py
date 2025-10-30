from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_sum = float('inf')
        for k in range(l, r + 1):
            for i in range(n - k + 1):
                current_sum = prefix[i + k] - prefix[i]
                if current_sum > 0:
                    min_sum = min(min_sum, current_sum)
        
        return min_sum if min_sum != float('inf') else -1