from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        target_score = nums[0] + nums[1]
        count = 0
        
        i = 0
        while i < len(nums) - 1:
            if nums[i] + nums[i + 1] == target_score:
                count += 1
                i += 2
            else:
                break
        
        return count