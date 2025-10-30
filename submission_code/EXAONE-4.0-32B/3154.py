from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        left_max = [0] * n
        left_max[0] = 0
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], nums[i-1])
        
        right_max = [0] * n
        right_max[n-1] = 0
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], nums[i+1])
        
        ans = 0
        for j in range(1, n-1):
            candidate = (left_max[j] - nums[j]) * right_max[j]
            if candidate > ans:
                ans = candidate
                
        return ans