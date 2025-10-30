from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        current_ops = 0
        
        for i in range(n):
            current_ops += diff[i]
            rem = nums[i] - current_ops
            
            if rem < 0:
                return False
            
            if rem > 0:
                if i + k > n:
                    return False
                current_ops += rem
                diff[i + k] -= rem
        
        return True