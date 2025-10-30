from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        """
        Dynamic programming with two resources.
        For every element we decide whether to:
            - do nothing                 (consume 0 op1, 0 op2)
            - use only operation-1       (consume 1 op1, 0 op2)
            - use only operation-2       (consume 0 op1, 1 op2)   – only if value ≥ k
            - use both operations        (consume 1 op1, 1 op2)   – only if feasible
        dp[i][j] = minimal possible partial sum after already spending
                   i uses of operation-1 and j uses of operation-2.
        """
        n = len(nums)
        INF = 10**18
        # dp[o1][o2]  (0-initialised with +inf except dp[0][0] = 0)
        dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0

        for val in nums:
            # list of feasible (extra_op1, extra_op2, resulting_value)
            choices = []

            # 1) do nothing
            choices.append((0, 0, val))

            # 2) only operation-1
            half = (val + 1) // 2
            choices.append((1, 0, half))

            # 3) only / also operation-2  (needs val ≥ k at the moment of applying it)
            if val >= k:
                choices.append((0, 1, val - k))

                # both operations – choose the better order
                best_after_both = (val - k + 1) // 2          # op2 first, then op1
                if half >= k:
                    best_after_both = min(best_after_both, half - k)   # op1 first, then op2
                choices.append((1, 1, best_after_both))

            # transition to next dp
            nxt = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
            for i in range(op1 + 1):
                for j in range(op2 + 1):
                    if dp[i][j] == INF:
                        continue
                    base = dp[i][j]
                    for add1, add2, cost in choices:
                        ni = i + add1
                        nj = j + add2
                        if ni <= op1 and nj <= op2:
                            nxt[ni][nj] = min(nxt[ni][nj], base + cost)
            dp = nxt

        # answer – we may use fewer than the allowed operations
        return min(min(row) for row in dp)