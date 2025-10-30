from typing import List
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        memo = {}
        
        def solve(index, current_nums, op1_remaining, op2_remaining):
            if index == len(current_nums):
                return sum(current_nums)
            state = (index, tuple(current_nums), op1_remaining, op2_remaining)
            if state in memo:
                return memo[state]
            
            res = float('inf')
            
            # Option 1: No operation
            res = min(res, solve(index + 1, list(current_nums), op1_remaining, op2_remaining))
            
            # Option 2: Operation 1
            if op1_remaining > 0:
                temp_nums = list(current_nums)
                temp_nums[index] = (temp_nums[index] + 1) // 2
                res = min(res, solve(index + 1, temp_nums, op1_remaining - 1, op2_remaining))
                
            # Option 3: Operation 2
            if op2_remaining > 0 and current_nums[index] >= k:
                temp_nums = list(current_nums)
                temp_nums[index] = max(0, temp_nums[index] - k)
                res = min(res, solve(index + 1, temp_nums, op1_remaining, op2_remaining - 1))
                
            # Option 4: Operation 1 and Operation 2
            if op1_remaining > 0 and op2_remaining > 0 and current_nums[index] >= k:
                # Order 1: Op1 then Op2
                temp_nums1 = list(current_nums)
                temp_val1 = (temp_nums1[index] + 1) // 2
                temp_nums1[index] = max(0, temp_val1 - k)
                res = min(res, solve(index + 1, temp_nums1, op1_remaining - 1, op2_remaining - 1))
                
                # Order 2: Op2 then Op1
                temp_nums2 = list(current_nums)
                temp_val2 = max(0, temp_nums2[index] - k)
                temp_nums2[index] = (temp_val2 + 1) // 2
                res = min(res, solve(index + 1, temp_nums2, op1_remaining - 1, op2_remaining - 1))
                
            memo[state] = res
            return res
            
        return solve(0, nums, op1, op2)