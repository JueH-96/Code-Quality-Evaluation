import math
from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        infinity = float('inf')
        
        # dp[j][l] will store the minimum sum considering elements processed so far,
        # using at most j Op1 operations and at most l Op2 operations.
        # Size: (op1 + 1) x (op2 + 1)
        dp = [[infinity for _ in range(op2 + 1)] for _ in range(op1 + 1)]

        # Base case: sum is 0 with 0 elements processed and 0 operations used
        dp[0][0] = 0

        # Iterate through each number in nums
        for current_num in nums:
            
            # Pre-calculate values after potential operations
            # Operation 1: Divide by 2, round up
            val_op1 = math.ceil(current_num / 2)
            # Operation 2: Subtract k, must be >= k, result non-negative
            val_op2 = max(0, current_num - k)
            # Both Operations: Op1 then Op2, result non-negative
            val_both = max(0, math.ceil(current_num / 2) - k)

            # Iterate through the DP table backwards
            # dp[j][l] currently stores the minimum sum using elements processed *before* current_num,
            # with at most j Op1s and at most l Op2s.
            # We update it to represent the state using elements *up to and including* current_num.
            # Iterating backwards ensures that dp[j-1][l], dp[j][l-1], and dp[j-1][l-1] still hold
            # the values from the *previous* element's calculation when we calculate dp[j][l].
            for j in range(op1, -1, -1):
                for l in range(op2, -1, -1):
                    
                    # If the current state dp[j][l] was unreachable using previous elements,
                    # we cannot apply any operation (including no-op) on current_num starting from this state.
                    if dp[j][l] == infinity:
                         continue

                    # Option 1: Don't apply any operation to current_num.
                    # The new state (j, l) using elements 0..i (with at most j Op1s, at most l Op2s)
                    # can be reached from the old state (j, l) using elements 0..i-1 (with at most j Op1s, at most l Op2s).
                    # The cost is old_dp[j][l] + current_num.
                    # Update the current cell dp[j][l] with this candidate sum.
                    # Since we are iterating backwards, dp[j][l] still holds the old value here.
                    dp[j][l] = dp[j][l] + current_num # This is the contribution if we don't operate current_num

                    # Option 2: Apply Op1 to current_num.
                    # The new state (j, l) using elements 0..i (with at most j Op1s, at most l Op2s)
                    # can be reached from the old state (j-1, l) using elements 0..i-1 (with at most j-1 Op1s, at most l Op2s).
                    # This is only possible if j > 0 (meaning we have capacity for at least 1 Op1).
                    # The cost is old_dp[j-1][l] + val_op1.
                    # Update dp[j][l] with the minimum of its current value and this candidate sum.
                    # Since we are iterating backwards, dp[j-1][l] still holds the old value here.
                    if j > 0:
                        dp[j][l] = min(dp[j][l], dp[j-1][l] + val_op1)

                    # Option 3: Apply Op2 to current_num.
                    # The new state (j, l) can be reached from the old state (j, l-1).
                    # This is only possible if l > 0 (meaning we have capacity for at least 1 Op2).
                    # Cost = old_dp[j][l-1] + val_op2.
                    if l > 0:
                        dp[j][l] = min(dp[j][l], dp[j][l-1] + val_op2)

                    # Option 4: Apply Both ops (Op1 then Op2) to current_num.
                    # The new state (j, l) can be reached from the old state (j-1, l-1).
                    # This is only possible if j > 0 and l > 0 (meaning we have capacity for at least 1 Op1 and 1 Op2).
                    # Cost = old_dp[j-1][l-1] + val_both.
                    if j > 0 and l > 0:
                        dp[j][l] = min(dp[j][l], dp[j-1][l-1] + val_both)

        # After iterating through all numbers, dp[op1][op2] holds the minimum sum
        # using all elements with at most op1 Op1s and at most op2 Op2s.
        return dp[op1][op2]