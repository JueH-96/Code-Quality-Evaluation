from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Total sum before any operations
        total = sum(nums)
        # dp[u][v] = maximum total reduction using u op1-uses and v op2-uses so far
        # We initialize with -inf for unreachable states, except dp[0][0] = 0
        INF = 10**18
        dp = [[-INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for num in nums:
            # Precompute single-op reductions
            half_up = (num + 1) // 2
            r_op1 = num - half_up        # reduction if we do op1 only
            can_op2 = (num >= k)
            if can_op2:
                r_op2 = k               # reduction if we do op2 only
                half_after_op2 = (num - k + 1) // 2
                r_op2_then_op1 = num - half_after_op2
            # for op1 then op2
            r_op1_then_op2 = r_op1 + k
            cond1 = (half_up >= k)       # can do op1->op2
            cond2 = can_op2              # can do op2->op1
            
            # Next-layer DP
            dp2 = [[-INF] * (op2 + 1) for _ in range(op1 + 1)]
            
            for u in range(op1 + 1):
                for v in range(op2 + 1):
                    if dp[u][v] < 0:
                        continue
                    base = dp[u][v]
                    # 1) no operation on this element
                    if base > dp2[u][v]:
                        dp2[u][v] = base
                    # 2) only op1
                    if u + 1 <= op1:
                        val = base + r_op1
                        if val > dp2[u + 1][v]:
                            dp2[u + 1][v] = val
                    # 3) only op2
                    if v + 1 <= op2 and can_op2:
                        val = base + r_op2
                        if val > dp2[u][v + 1]:
                            dp2[u][v + 1] = val
                    # 4) both operations
                    if u + 1 <= op1 and v + 1 <= op2:
                        best_r = -INF
                        if cond1:
                            best_r = max(best_r, r_op1_then_op2)
                        if cond2:
                            best_r = max(best_r, r_op2_then_op1)
                        if best_r >= 0:
                            val = base + best_r
                            if val > dp2[u + 1][v + 1]:
                                dp2[u + 1][v + 1] = val
            
            dp = dp2
        
        # Find the best (maximum) reduction achievable within op1/op2 budgets
        best_reduction = 0
        for u in range(op1 + 1):
            for v in range(op2 + 1):
                if dp[u][v] > best_reduction:
                    best_reduction = dp[u][v]
        
        # Minimum possible sum = original sum - best reduction
        return total - best_reduction