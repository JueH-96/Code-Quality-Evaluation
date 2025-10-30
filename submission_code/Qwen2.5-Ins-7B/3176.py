from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        left_min = [0] * n
        right_min = [0] * n
        min_val = float('inf')
        
        left_min[0] = nums[0]
        for i in range(1, n):
            left_min[i] = min(left_min[i-1], nums[i])
        
        right_min[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        for j in range(1, n-1):
            if left_min[j-1] < nums[j] > right_min[j+1]:
                min_val = min(min_val, left_min[j-1] + nums[j] + right_min[j+1])
        
        return min_val if min_val != float('inf') else -1