from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changes = 0
        
        for i in range(n // 2):
            left = nums[i]
            right = nums[n - i - 1]
            if left != right:
                changes += 1
        
        return changes