from math import ceil
from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        total = sum(nums)
        # dp[i][a][b] = maximum total reduction we can achieve using the first i elements
        # with a used op1 operations and b used op2 operations.
        # Initialize with -infinity for impossible states.
        dp = [[[-10**9 for _ in range(op2+1)] for _ in range(op1+1)] for _ in range(n+1)]
        dp[0][0][0] = 0
        
        for i in range(n):
            x = nums[i]
            # Calculate possible benefits for different operation choices for this element.
            # Option 0: no operation.
            cost0 = (0, 0)
            benefit0 = 0
            
            # Option 1: op1 only. Always allowed.
            cost1 = (1, 0)
            benefit1 = x - ceil(x/2)
            
            # Option 2: op2 only. Allowed if x >= k.
            if x >= k:
                cost2 = (0, 1)
                benefit2 = k  # subtract k reduces that element by k.
            else:
                cost2 = None
            
            # Option 3: both operations.
            # Two possible orders:
            # Order A: op1 then op2, but allowed only if value after op1 is >= k.
            #    result = ceil(x/2)-k, reduction = x - (ceil(x/2)-k) = (x - ceil(x/2)) + k.
            # Order B: op2 then op1, allowed if x >= k.
            #    result = ceil((x-k)/2), reduction = x - ceil((x-k)/2).
            best_both = -10**9
            if x >= k:
                # Check Order A condition.
                opA = -10**9
                half_val = ceil(x/2)
                if half_val >= k:
                    opA = (x - half_val) + k
                opB = x - ceil((x - k)/2)
                best_both = max(opA, opB)
                cost3 = (1, 1)
            else:
                cost3 = None
            
            # Now update dp for index i.
            for a in range(op1+1):
                for b in range(op2+1):
                    if dp[i][a][b] < 0:
                        continue
                    cur = dp[i][a][b]
                    # Option 0: do nothing.
                    if dp[i+1][a][b] < cur:
                        dp[i+1][a][b] = cur
                    # Option 1: use op1 only, if available.
                    na, nb = a + cost1[0], b + cost1[1]
                    if na <= op1 and nb <= op2:
                        new_val = cur + benefit1
                        if new_val > dp[i+1][na][nb]:
                            dp[i+1][na][nb] = new_val
                    # Option 2: use op2 only if allowed.
                    if cost2 is not None:
                        na, nb = a + cost2[0], b + cost2[1]
                        if na <= op1 and nb <= op2:
                            new_val = cur + benefit2
                            if new_val > dp[i+1][na][nb]:
                                dp[i+1][na][nb] = new_val
                    # Option 3: use both if allowed.
                    if cost3 is not None:
                        na, nb = a + cost3[0], b + cost3[1]
                        if na <= op1 and nb <= op2:
                            new_val = cur + best_both
                            if new_val > dp[i+1][na][nb]:
                                dp[i+1][na][nb] = new_val
        
        best = 0
        for a in range(op1+1):
            for b in range(op2+1):
                if dp[n][a][b] > best:
                    best = dp[n][a][b]
        return total - best