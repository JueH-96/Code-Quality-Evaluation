from typing import List
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        INF = float('inf')
        # dp[i][u][v]: minimal sum for first i elements with u op1 used and v op2 used.
        dp = [[[INF]*(op2+1) for _ in range(op1+1)] for _ in range(n+1)]
        dp[0][0][0] = 0
        
        for i in range(n):
            a = nums[i]
            # Calculate the cost if we apply operations on a:
            # Option0: no operation.
            cost0 = a
            
            # Option1: apply only op1 (divide by 2, rounding up)
            cost1 = math.ceil(a/2)
            
            # Option2: apply only op2 (subtract k) if allowed.
            can_op2 = (a >= k)
            cost2 = a - k if can_op2 else None
            
            # Option3: apply both operations.
            # There are two orders:
            cost_both = INF
            # Order: op1 then op2. Allowed if after op1, value is >= k.
            temp1 = math.ceil(a/2)
            if temp1 >= k:
                cost_both = min(cost_both, temp1 - k)
            # Order: op2 then op1. Allowed if a >= k.
            if a >= k:
                temp2 = math.ceil((a - k)/2)
                cost_both = min(cost_both, temp2)
            # If no valid order, then cost_both remains INF (i.e. not allowed)
            
            for used_op1 in range(op1+1):
                for used_op2 in range(op2+1):
                    if dp[i][used_op1][used_op2] == INF:
                        continue
                    cur = dp[i][used_op1][used_op2]
                    # Option 0: no operation
                    if cur + cost0 < dp[i+1][used_op1][used_op2]:
                        dp[i+1][used_op1][used_op2] = cur + cost0
                    # Option 1: use op1 only if available (one op1 used)
                    if used_op1 < op1:
                        if cur + cost1 < dp[i+1][used_op1+1][used_op2]:
                            dp[i+1][used_op1+1][used_op2] = cur + cost1
                    # Option 2: use op2 only if available and allowed.
                    if used_op2 < op2 and can_op2:
                        if cur + cost2 < dp[i+1][used_op1][used_op2+1]:
                            dp[i+1][used_op1][used_op2+1] = cur + cost2
                    # Option 3: use both operations at the same index if available and allowed.
                    if used_op1 < op1 and used_op2 < op2 and cost_both != INF:
                        if cur + cost_both < dp[i+1][used_op1+1][used_op2+1]:
                            dp[i+1][used_op1+1][used_op2+1] = cur + cost_both
        
        # find the minimal sum with at most op1 and op2 used.
        ans = INF
        for u in range(op1+1):
            for v in range(op2+1):
                ans = min(ans, dp[n][u][v])
        return ans