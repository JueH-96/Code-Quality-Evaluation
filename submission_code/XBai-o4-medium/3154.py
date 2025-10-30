from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_left = [0] * n
        current_max = 0
        for i in range(1, n):
            current_max = max(current_max, nums[i-1])
            max_left[i] = current_max
        
        max_right = [0] * n
        current_max = 0
        for i in range(n-2, -1, -1):
            current_max = max(current_max, nums[i+1])
            max_right[i] = current_max
        
        max_val = 0
        for j in range(1, n-1):
            current = (max_left[j] - nums[j]) * max_right[j]
            if current > max_val:
                max_val = current
        
        return max(max_val, 0)