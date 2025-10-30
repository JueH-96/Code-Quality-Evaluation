from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        total_ops = 0
        prev = target[0] - nums[0]
        total_ops += max(0, prev)
        
        for i in range(1, n):
            curr = target[i] - nums[i]
            diff = curr - prev
            if diff > 0:
                total_ops += diff
            prev = curr
            
        if -prev > 0:
            total_ops += -prev
            
        return total_ops