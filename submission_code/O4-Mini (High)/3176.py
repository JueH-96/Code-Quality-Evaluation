from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('inf')
        
        for j in range(1, n-1):
            # find smallest nums[i] < nums[j] for i < j
            left_min = float('inf')
            for i in range(j):
                if nums[i] < nums[j] and nums[i] < left_min:
                    left_min = nums[i]
            if left_min == float('inf'):
                continue
            
            # find smallest nums[k] < nums[j] for k > j
            right_min = float('inf')
            for k in range(j+1, n):
                if nums[k] < nums[j] and nums[k] < right_min:
                    right_min = nums[k]
            if right_min == float('inf'):
                continue
            
            # update best sum
            total = left_min + nums[j] + right_min
            if total < best:
                best = total
        
        return best if best < float('inf') else -1