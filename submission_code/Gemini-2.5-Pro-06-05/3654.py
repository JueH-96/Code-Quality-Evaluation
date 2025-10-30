from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        """
        Calculates the minimum possible sum of an array after applying a limited number of two types of operations.

        This problem is solved using dynamic programming. The state of our DP is defined by the number of
        operations of each type used. Let dp[c1][c2] be the minimum sum of the prefix of `nums`
        processed so far, using exactly c1 of Operation 1 and c2 of Operation 2.

        We iterate through each number in `nums` and update the DP table. For each number, we consider
        four possibilities: applying no operation, only Operation 1, only Operation 2, or both.
        Each choice transitions from a previous DP state (from the last number) to a new one.

        To optimize space, we only need to store the DP table for the previous number to calculate
        the table for the current one.
        """
        
        # dp[c1][c2] represents the minimum sum after processing a prefix of nums,
        # using c1 op1's and c2 op2's.
        dp = [[float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0

        for num in nums:
            new_dp = [[float('inf')] * (op2 + 1) for _ in range(op1 + 1)]

            # Calculate potential resulting values for the current 'num'
            
            # v0: result of no operation
            v0 = num
            
            # v1: result of Operation 1 (divide by 2, round up)
            v1 = (num + 1) // 2
            
            # v2: result of Operation 2 (subtract k)
            v2 = num - k if num >= k else float('inf')
            
            # v3: result of applying both operations. We take the minimum of the two possible orders.
            v3 = float('inf')
            # Order A: Op1 then Op2
            res_op1 = (num + 1) // 2
            val_1_then_2 = res_op1 - k if res_op1 >= k else float('inf')
            # Order B: Op2 then Op1
            val_2_then_1 = float('inf')
            if num >= k:
                res_op2 = num - k
                val_2_then_1 = (res_op2 + 1) // 2
            v3 = min(val_1_then_2, val_2_then_1)

            # "Push" style DP: iterate through source states and update destination states
            for c1 in range(op1 + 1):
                for c2 in range(op2 + 1):
                    if dp[c1][c2] == float('inf'):
                        continue
                    
                    prev_sum = dp[c1][c2]
                    
                    # Choice 1: No operation on `num`.
                    new_dp[c1][c2] = min(new_dp[c1][c2], prev_sum + v0)
                    
                    # Choice 2: Operation 1 on `num`. Uses one op1.
                    if c1 + 1 <= op1:
                        new_dp[c1 + 1][c2] = min(new_dp[c1 + 1][c2], prev_sum + v1)
            
                    # Choice 3: Operation 2 on `num`. Uses one op2.
                    if c2 + 1 <= op2 and v2 != float('inf'):
                        new_dp[c1][c2 + 1] = min(new_dp[c1][c2 + 1], prev_sum + v2)
            
                    # Choice 4: Both operations on `num`. Uses one op1 and one op2.
                    if c1 + 1 <= op1 and c2 + 1 <= op2 and v3 != float('inf'):
                        new_dp[c1 + 1][c2 + 1] = min(new_dp[c1 + 1][c2 + 1], prev_sum + v3)

            # Update dp table for the next number
            dp = new_dp
    
        # The result is the minimum value in the final DP table
        min_total_sum = float('inf')
        for row in dp:
            min_total_sum = min(min_total_sum, min(row))
            
        return int(min_total_sum)