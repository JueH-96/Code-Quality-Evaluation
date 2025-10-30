from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        max_i_before = [0] * n
        max_so_far = nums[0]
        max_i_before[0] = 0  # Not used as j starts from 1
        
        for j in range(1, n):
            max_i_before[j] = max_so_far
            if nums[j] > max_so_far:
                max_so_far = nums[j]
        
        best_j = [0] * n
        for j in range(1, n):
            best_j[j] = max_i_before[j] - nums[j]
        
        max_best_upto = [float('-inf')] * n
        current_max = float('-inf')
        for j in range(1, n):
            current_val = best_j[j]
            if current_val > current_max:
                current_max = current_val
            max_best_upto[j] = current_max
        
        max_val = float('-inf')
        for k in range(2, n):
            current_max_best = max_best_upto[k-1]
            current_value = current_max_best * nums[k]
            if current_value > max_val:
                max_val = current_value
        
        return max_val if max_val > 0 else 0