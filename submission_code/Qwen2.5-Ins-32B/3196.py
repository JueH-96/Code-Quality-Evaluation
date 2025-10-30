from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_score = 0
        current_cost = 0
        
        for right in range(len(nums)):
            if right > 0:
                current_cost += (nums[right] - nums[right - 1]) * (right - left)
            
            while current_cost > k:
                current_cost -= (nums[right] - nums[left])
                left += 1
            
            max_score = max(max_score, right - left + 1)
        
        return max_score