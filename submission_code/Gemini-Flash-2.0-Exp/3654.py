from typing import List
import heapq

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        def calculate_sum(arr):
            return sum(arr)
        
        import copy
        
        min_sum = float('inf')
        
        
        
        def solve(current_nums, current_op1, current_op2):
            nonlocal min_sum
            
            if current_op1 == 0 and current_op2 == 0:
                min_sum = min(min_sum, calculate_sum(current_nums))
                return
            
            
            if current_op1 > 0:
                for i in range(n):
                    temp_nums = copy.deepcopy(current_nums)
                    temp_nums[i] = (temp_nums[i] + 1) // 2
                    solve(temp_nums, current_op1 - 1, current_op2)
            
            if current_op2 > 0:
                for i in range(n):
                    temp_nums = copy.deepcopy(current_nums)
                    if temp_nums[i] >= k:
                        temp_nums[i] -= k
                        solve(temp_nums, current_op1, current_op2 - 1)
            
            if current_op1 == 0 and current_op2 > 0:
                min_sum = min(min_sum, calculate_sum(current_nums))
                return
            if current_op2 == 0 and current_op1 > 0:
                min_sum = min(min_sum, calculate_sum(current_nums))
                return
        
        solve(nums, op1, op2)
        
        return min_sum