from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        import math
        
        original_sum = sum(nums)
        
        # Initialize DP table with zeros
        dp = [[0] * (op2 + 1) for _ in range(op1 + 1)]
        
        for num in nums:
            options = []
            if num >= k:
                # Option 1: Operation 1 only
                reduction1 = num - math.ceil(num / 2)
                options.append((1, 0, reduction1))
                
                # Option 2: Operation 2 only
                reduction2 = k
                options.append((0, 1, reduction2))
                
                # Option 3: Both operations
                reduction_both = num - math.ceil((num - k) / 2)
                options.append((1, 1, reduction_both))
                
                # Option 4: No operations
                options.append((0, 0, 0))
            else:
                # Option 1: Operation 1 only
                reduction1 = num - math.ceil(num / 2)
                options.append((1, 0, reduction1))
                
                # Option 2: No operations
                options.append((0, 0, 0))
            
            # Update DP table for current number
            next_dp = [[0] * (op2 + 1) for _ in range(op1 + 1)]
            for j in range(op1 + 1):
                for l in range(op2 + 1):
                    for cost_op1, cost_op2, reduction in options:
                        if j >= cost_op1 and l >= cost_op2:
                            next_dp[j][l] = max(next_dp[j][l], dp[j - cost_op1][l - cost_op2] + reduction)
                    # Also consider not choosing any operation for this element
                    if j + l > 0:
                        next_dp[j][l] = max(next_dp[j][l], dp[j][l])
            dp = next_dp
        
        # The maximum reduction is in dp[op1][op2]
        max_reduction = dp[op1][op2]
        min_sum = original_sum - max_reduction
        return min_sum