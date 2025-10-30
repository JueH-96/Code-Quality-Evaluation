from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        left_max = [-float('inf')] * n
        left_min = [float('inf')] * n
        
        for j in range(1, n):
            left_max[j] = max(left_max[j-1], nums[j-1])
            left_min[j] = min(left_min[j-1], nums[j-1])
        
        right_max = [-float('inf')] * n
        right_min = [float('inf')] * n
        
        for j in range(n-2, -1, -1):
            right_max[j] = max(nums[j+1], right_max[j+1])
            right_min[j] = min(nums[j+1], right_min[j+1])
        
        global_max = -float('inf')
        
        for j in range(1, n-1):
            current_b = nums[j]
            lm = left_max[j]
            lmin = left_min[j]
            rm = right_max[j]
            rmin = right_min[j]
            
            val1 = (lm - current_b) * rm
            val2 = (lm - current_b) * rmin
            val3 = (lmin - current_b) * rm
            val4 = (lmin - current_b) * rmin
            
            current_max = max(val1, val2, val3, val4)
            if current_max > global_max:
                global_max = current_max
        
        return max(0, global_max)