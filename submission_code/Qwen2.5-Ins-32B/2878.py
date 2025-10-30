from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        for i in range(n):
            nums[i] -= diff[i]
            if nums[i] < 0:
                return False
            if i + k <= n:
                diff[i + k] += nums[i]
        
        return all(x == 0 for x in nums)