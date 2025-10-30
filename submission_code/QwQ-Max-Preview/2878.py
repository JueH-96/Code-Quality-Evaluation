from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        current_ops = 0
        
        for i in range(n):
            current_ops += diff[i]
            needed = nums[i] - current_ops
            
            if needed < 0:
                return False
            
            if needed > 0:
                if i + k > n:
                    return False
                current_ops += needed
                if i + k < n:
                    diff[i + k] -= needed
        
        return True