from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        target_sum = nums[0] + nums[1]
        count = 1
        i = 2
        
        while i + 1 < n:
            current_sum = nums[i] + nums[i + 1]
            if current_sum == target_sum:
                count += 1
                i += 2
            else:
                break
        
        return count