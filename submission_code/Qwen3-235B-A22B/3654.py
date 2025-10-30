from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        original_sum = sum(nums)
        n = len(nums)
        
        # Initialize DP table with -infinity to represent impossible states
        dp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0  # Base case: no operations used
        
        for x in nums:
            temp_dp = [row[:] for row in dp]  # Create a copy of current DP state
            options = []
            
            # Option 0: do nothing
            options.append((0, 0, 0))
            
            # Option 1: apply op1 (divide by 2, ceil)
            ceil_div = (x + 1) // 2
            gain_op1 = x - ceil_div
            options.append((1, 0, gain_op1))
            
            # Option 2: apply op2 only
            if x >= k:
                gain_op2 = k
                options.append((0, 1, gain_op2))
            
            # Option 3: apply op1 followed by op2
            ceil_div = (x + 1) // 2
            if ceil_div >= k:
                gain_combined = (x - ceil_div) + k
                options.append((1, 1, gain_combined))
            
            # Option 4: apply op2 followed by op1
            if x >= k:
                after_op2 = x - k
                ceil_after = (after_op2 + 1) // 2
                gain_combined = x - ceil_after
                options.append((1, 1, gain_combined))
            
            # Update temp_dp with all possible options for current element
            for op1_used, op2_used, gain in options:
                for p in range(op1, -1, -1):
                    for q in range(op2, -1, -1):
                        prev_p = p - op1_used
                        prev_q = q - op2_used
                        if prev_p >= 0 and prev_q >= 0:
                            if dp[prev_p][prev_q] + gain > temp_dp[p][q]:
                                temp_dp[p][q] = dp[prev_p][prev_q] + gain
            
            # Assign temp_dp back to dp for the next iteration
            dp = temp_dp
        
        # Find the maximum gain possible within allowed operations
        max_gain = 0
        for p in range(op1 + 1):
            for q in range(op2 + 1):
                if dp[p][q] > max_gain:
                    max_gain = dp[p][q]
        
        return original_sum - max_gain