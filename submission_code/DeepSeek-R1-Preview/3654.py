from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        original_sum = sum(nums)
        n = len(nums)
        options_list = []
        for x in nums:
            options = []
            # Option 0: do nothing
            options.append((0, 0, 0))
            
            # Option 1: apply op1 only
            new_val = (x + 1) // 2
            options.append((1, 0, x - new_val))
            
            # Option 2: apply op2 only (if possible)
            if x >= k:
                new_val = x - k
                options.append((0, 1, x - new_val))
            
            # Option 3: apply op1 then op2 (if possible after op1)
            op1_val = (x + 1) // 2
            if op1_val >= k:
                new_val = op1_val - k
                reduction = x - new_val
                options.append((1, 1, reduction))
            
            # Option 4: apply op2 then op1 (if op2 is possible)
            if x >= k:
                op2_val = x - k
                op1_val = (op2_val + 1) // 2
                reduction = x - op1_val
                options.append((1, 1, reduction))
            
            options_list.append(options)
        
        # Initialize DP table
        dp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        # Process each index
        for i in range(n):
            # Create a temporary DP to store new states
            temp_dp = [row.copy() for row in dp]
            for a_op, b_op, r_op in options_list[i]:
                # Update temp_dp by considering this option
                for a in range(op1, a_op - 1, -1):
                    for b in range(op2, b_op - 1, -1):
                        prev_a = a - a_op
                        prev_b = b - b_op
                        if prev_a >= 0 and prev_b >= 0 and dp[prev_a][prev_b] != -float('inf'):
                            if temp_dp[a][b] < dp[prev_a][prev_b] + r_op:
                                temp_dp[a][b] = dp[prev_a][prev_b] + r_op
            # Update dp to the new state
            dp = temp_dp
        
        # Find the maximum reduction possible
        max_reduction = 0
        for a in range(op1 + 1):
            for b in range(op2 + 1):
                if dp[a][b] > max_reduction:
                    max_reduction = dp[a][b]
        
        return original_sum - max_reduction