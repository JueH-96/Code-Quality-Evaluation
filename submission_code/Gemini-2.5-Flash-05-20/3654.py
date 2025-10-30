import math
from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        total_original_sum = sum(nums)

        # dp[j][l] will store the maximum reduction achievable using
        # j operations of type 1 and l operations of type 2.
        # Initialize with -float('inf') to represent unreachable states.
        dp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0  # Base case: 0 operations, 0 reduction is always possible

        for val in nums:
            # Calculate potential reductions for the current number 'val'
            # Use -float('inf') if an operation is not possible for 'val'.

            # Reduction from Op1 only: val - ceil(val / 2)
            # For non-negative integer val, ceil(val / 2) is (val + 1) // 2
            val_after_op1_temp = (val + 1) // 2
            reduction_op1 = val - val_after_op1_temp

            # Reduction from Op2 only: k, if val >= k
            reduction_op2 = k if val >= k else -float('inf')

            # Reduction from Op1 then Op2: val - (ceil(val / 2) - k)
            # Requires ceil(val / 2) >= k
            reduction_op1_then_op2 = -float('inf')
            if val_after_op1_temp >= k:
                reduction_op1_then_op2 = val - (val_after_op1_temp - k)

            # Reduction from Op2 then Op1: val - ceil((val - k) / 2)
            # Requires val >= k
            reduction_op2_then_op1 = -float('inf')
            if val >= k:
                val_after_op2_temp = val - k
                reduction_op2_then_op1 = val - ((val_after_op2_temp + 1) // 2)

            # Iterate through dp table in reverse to simulate 0/1 knapsack behavior.
            # This ensures that when we update dp[j_new][l_new],
            # dp[j][l] (used on the RHS) refers to the state *before* considering the current 'val'.
            for j in range(op1, -1, -1):
                for l in range(op2, -1, -1):
                    # If the current dp state (dp[j][l]) is unreachable, skip as no new reduction can be built from it.
                    if dp[j][l] == -float('inf'):
                        continue

                    # Try to apply Option 1: Op1 only to current 'val'
                    # If we have budget for one more Op1 (j+1 <= op1)
                    if j + 1 <= op1:
                        dp[j+1][l] = max(dp[j+1][l], dp[j][l] + reduction_op1)

                    # Try to apply Option 2: Op2 only to current 'val'
                    # If we have budget for one more Op2 (l+1 <= op2) and Op2 is possible for 'val'
                    if l + 1 <= op2 and reduction_op2 != -float('inf'):
                        dp[j][l+1] = max(dp[j][l+1], dp[j][l] + reduction_op2)

                    # Try to apply Option 3: Op1 then Op2 to current 'val'
                    # If we have budget for both (j+1 <= op1, l+1 <= op2) and this combo is possible
                    if j + 1 <= op1 and l + 1 <= op2 and reduction_op1_then_op2 != -float('inf'):
                        dp[j+1][l+1] = max(dp[j+1][l+1], dp[j][l] + reduction_op1_then_op2)

                    # Try to apply Option 4: Op2 then Op1 to current 'val'
                    # If we have budget for both (j+1 <= op1, l+1 <= op2) and this combo is possible
                    if j + 1 <= op1 and l + 1 <= op2 and reduction_op2_then_op1 != -float('inf'):
                        dp[j+1][l+1] = max(dp[j+1][l+1], dp[j][l] + reduction_op2_then_op1)
        
        # After processing all numbers, find the maximum reduction achievable
        # using at most op1 and op2 operations.
        # This means checking all states within the budget limits.
        max_reduction = 0
        for j in range(op1 + 1):
            for l in range(op2 + 1):
                max_reduction = max(max_reduction, dp[j][l])
        
        # The minimum possible sum is the original sum minus the maximum reduction.
        return total_original_sum - max_reduction