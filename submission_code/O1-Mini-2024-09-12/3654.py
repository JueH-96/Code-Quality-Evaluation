from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Initialize DP array
        prev = [[float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        prev[0][0] = 0
        
        for num in nums:
            current = [[float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
            for used_op1 in range(op1 + 1):
                for used_op2 in range(op2 + 1):
                    if prev[used_op1][used_op2] == float('inf'):
                        continue
                    current_val = prev[used_op1][used_op2]
                    
                    # Option 0: Do nothing
                    new_sum = current_val + num
                    if new_sum < current[used_op1][used_op2]:
                        current[used_op1][used_op2] = new_sum
                    
                    # Option 1: Apply Operation 1
                    if used_op1 < op1:
                        new_used_op1 = used_op1 + 1
                        reduced = (num + 1) // 2
                        new_sum = current_val + reduced
                        if new_sum < current[new_used_op1][used_op2]:
                            current[new_used_op1][used_op2] = new_sum
                            
                    # Option 2: Apply Operation 2
                    if num >= k and used_op2 < op2:
                        new_used_op2 = used_op2 + 1
                        reduced = num - k
                        new_sum = current_val + reduced
                        if new_sum < current[used_op1][new_used_op2]:
                            current[used_op1][new_used_op2] = new_sum
                            
                    # Option 3: Apply both Operation 1 and Operation 2
                    if num >= k and used_op1 < op1 and used_op2 < op2:
                        # Apply Op1 then Op2
                        option_a = ((num + 1) // 2) - k
                        # Apply Op2 then Op1
                        option_b = ((num - k) + 1) // 2
                        min_val = min(option_a, option_b)
                        min_val = max(min_val, 0)
                        new_used_op1 = used_op1 + 1
                        new_used_op2 = used_op2 + 1
                        new_sum = current_val + min_val
                        if new_sum < current[new_used_op1][new_used_op2]:
                            current[new_used_op1][new_used_op2] = new_sum
                            
            prev = current
        
        # Find the minimum sum after all operations
        min_sum = float('inf')
        for used_op1 in range(op1 + 1):
            for used_op2 in range(op2 + 1):
                if prev[used_op1][used_op2] < min_sum:
                    min_sum = prev[used_op1][used_op2]
        
        return min_sum