from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_prev = [0] * n
        for j in range(1, n):
            max_prev[j] = max(max_prev[j-1], nums[j-1])
        
        max_suffix = [0] * n
        for j in range(n-2, -1, -1):
            max_suffix[j] = max(nums[j+1], max_suffix[j+1])
        
        max_val = -float('inf')
        for j in range(1, n-1):
            current = (max_prev[j] - nums[j]) * max_suffix[j]
            if current > max_val:
                max_val = current
        
        return max(max_val, 0)