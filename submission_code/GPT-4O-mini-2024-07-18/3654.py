from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Create a list to store the potential values after operations
        potential_values = []
        
        # Calculate the potential values after applying both operations
        for num in nums:
            # Operation 1: Divide by 2, rounding up
            if op1 > 0:
                new_value_op1 = (num + 1) // 2  # Round up division
                potential_values.append(new_value_op1)
            else:
                potential_values.append(num)
        
        # Sort potential values to apply operation 2 on the largest values first
        potential_values.sort(reverse=True)
        
        # Apply Operation 2: Subtract k if possible
        for i in range(len(potential_values)):
            if op2 > 0 and potential_values[i] >= k:
                potential_values[i] -= k
                op2 -= 1
        
        # Calculate the final sum after applying the operations
        final_sum = sum(potential_values)
        
        return final_sum