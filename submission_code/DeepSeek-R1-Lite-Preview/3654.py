from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Helper function for ceiling division
        def ceil_div(a, b):
            return (a + b - 1) // b
        
        # Calculate the original sum of nums
        original_sum = sum(nums)
        
        # Initialize DP table: (op1+1) x (op2+1)
        dp = [[0] * (op2 + 1) for _ in range(op1 + 1)]
        
        for num in nums:
            # List to hold possible options for this num
            options = []
            
            # Option 0: Do nothing
            options.append((0, 0, 0))
            
            # Option 1: Apply Operation 1
            reduction_op1 = num - ceil_div(num, 2)
            options.append((1, 0, reduction_op1))
            
            # Option 2: Apply Operation 2, if applicable
            if num >= k:
                reduction_op2 = k
                options.append((0, 1, reduction_op2))
            else:
                options.append((0, 1, 0))  # No reduction if num < k
            
            # Option 3: Apply both operations in the best order
            reduction_op3 = float('-inf')
            # Order 1: Operation 1 then Operation 2
            after_op1 = ceil_div(num, 2)
            if after_op1 >= k:
                reduction_op3 = num - (after_op1 - k)
            # Order 2: Operation 2 then Operation 1
            if num >= k:
                after_op2 = num - k
                reduction_temp = num - ceil_div(after_op2, 2)
                reduction_op3 = max(reduction_op3, reduction_temp)
            # If at least one order is possible
            if reduction_op3 != float('-inf'):
                options.append((1, 1, reduction_op3))
            else:
                # Skip Option 3 if neither order is possible
                continue
            
            # Update DP table in reverse order to prevent reuse within the same update
            for i in range(op1, -1, -1):
                for j in range(op2, -1, -1):
                    for op in options:
                        op1_used, op2_used, reduction = op
                        new_i = i + op1_used
                        new_j = j + op2_used
                        if new_i <= op1 and new_j <= op2:
                            dp[new_i][new_j] = max(dp[new_i][new_j], dp[i][j] + reduction)
        
        # Find the maximum total reduction
        max_reduction = 0
        for i in range(op1 + 1):
            for j in range(op2 + 1):
                max_reduction = max(max_reduction, dp[i][j])
        
        # Calculate the minimum possible sum
        min_sum = original_sum - max_reduction
        return min_sum