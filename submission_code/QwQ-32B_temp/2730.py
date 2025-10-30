from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
        
        suffix = [0] * n
        suffix[n - 1] = 0
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i + 1]
        
        max_or = 0
        for i in range(n):
            shifted = nums[i] << k
            current_or = (prefix[i] | shifted) | suffix[i]
            if current_or > max_or:
                max_or = current_or
        
        return max_or