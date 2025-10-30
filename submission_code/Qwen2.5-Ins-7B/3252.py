from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                if all(nums[k] < nums[k+1] for k in range(n) if k not in range(i, j+1)):
                    count += 1
                    break
        
        return count + 1