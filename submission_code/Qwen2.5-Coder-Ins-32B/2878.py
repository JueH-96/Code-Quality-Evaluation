from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        operations = [0] * (n + 1)
        
        current_operations = 0
        for i in range(n):
            current_operations += operations[i]
            nums[i] += current_operations
            
            if nums[i] < 0:
                return False
            
            if nums[i] > 0:
                if i + k > n:
                    return False
                current_operations += nums[i]
                operations[i + k] -= nums[i]
        
        return True