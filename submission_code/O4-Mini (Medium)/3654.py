from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        total = sum(nums)
        # Precompute for each index the possible (use_op1, use_op2, reduction)
        choices = []
        for v in nums:
            c = []
            # no operation
            c.append((0, 0, 0))
            # op1 only: halve, rounding up
            v1 = (v + 1) // 2
            red1 = v - v1
            if red1 > 0:
                c.append((1, 0, red1))
            # op2 only: subtract k if possible
            if v >= k and k > 0:
                red2 = k
                c.append((0, 1, red2))
            # both operations, best order
            best_final = None
            # first op1 then op2 (if allowed)
            if v1 >= k:
                final1 = v1 - k
                best_final = final1 if best_final is None else min(best_final, final1)
            # first op2 then op1 (if allowed)
            if v >= k:
                final2 = (v - k + 1) // 2
                best_final = final2 if best_final is None else min(best_final, final2)
            if best_final is not None:
                red_both = v - best_final
                if red_both > 0:
                    c.append((1, 1, red_both))
            choices.append(c)

        # dp[u][v] = max reduction using up to u op1's and v op2's
        # initialize with -inf
        NEG_INF = -10**18
        dp = [[NEG_INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0

        for i in range(n):
            dp_next = [[NEG_INF] * (op2 + 1) for _ in range(op1 + 1)]
            for used1 in range(op1 + 1):
                for used2 in range(op2 + 1):
                    if dp[used1][used2] < 0:
                        continue
                    base = dp[used1][used2]
                    # try all choices for index i
                    for add1, add2, red in choices[i]:
                        n1 = used1 + add1
                        n2 = used2 + add2
                        if n1 <= op1 and n2 <= op2:
                            val = base + red
                            if val > dp_next[n1][n2]:
                                dp_next[n1][n2] = val
            dp = dp_next

        # find maximum reduction
        max_red = 0
        for u in range(op1 + 1):
            for v in range(op2 + 1):
                if dp[u][v] > max_red:
                    max_red = dp[u][v]

        return total - max_red