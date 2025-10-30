from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        for j in range(1, n - 1):
            left_min = min(nums[:j])
            right_min = min(nums[j + 1:])
            
            if left_min < nums[j] and right_min < nums[j]:
                min_sum = min(min_sum, left_min + nums[j] + right_min)
        
        return min_sum if min_sum != float('inf') else -1