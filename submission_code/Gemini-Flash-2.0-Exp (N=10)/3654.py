import heapq
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        
        def calculate_sum(arr):
            return sum(arr)

        
        min_sum = float('inf')

        
        for i in range(1 << (n * 2)):
            temp_nums = list(nums)
            current_op1 = 0
            current_op2 = 0
            
            
            for j in range(n):
                
                op1_bit = (i >> (j * 2)) & 1
                op2_bit = (i >> (j * 2 + 1)) & 1
                
                if op1_bit == 1 and current_op1 < op1:
                    temp_nums[j] = math.ceil(temp_nums[j] / 2)
                    current_op1 += 1
                
                if op2_bit == 1 and current_op2 < op2 and temp_nums[j] >= k:
                    temp_nums[j] -= k
                    current_op2 += 1
            
            
            if current_op1 <= op1 and current_op2 <= op2:
                min_sum = min(min_sum, calculate_sum(temp_nums))

        return min_sum