from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n == 0:
            return True
        
        x = [0] * n
        x[0] = nums[0]
        if x[0] < 0:
            return False
        
        for i in range(1, k):
            if i >= n:
                break
            x[i] = nums[i] - nums[i-1]
            if x[i] < 0:
                return False
        
        for i in range(k, n):
            prev_x = x[i - k]
            current_diff = nums[i] - nums[i-1]
            x[i] = prev_x + current_diff
            if x[i] < 0:
                return False
        
        return True