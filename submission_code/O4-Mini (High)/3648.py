from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # Precompute the lower-bound for j2[t] and i3[t] to ensure reachability
        # lower[t] = max(t, n-1 - t)
        lower = [max(t, n - 1 - t) for t in range(n)]
        # Sum of fruits on the main diagonal collected by child1
        diag_sum = 0
        for i in range(n):
            diag_sum += fruits[i][i]
        # A large negative value for unreachable states
        NEG_INF = -10**18

        # DP for child2 (from (0, n-1) to (n-1, n-1))
        # dp2_prev[j] = max fruits collected up to step t-1, ending at column j
        dp2_prev = [NEG_INF] * n
        # t = 0, child2 starts at j = n-1
        j0 = lower[0]  # = n-1
        # reward at (0, j0) unless it's the diagonal cell (0,0)
        dp2_prev[j0] = fruits[0][j0] if j0 != 0 else 0

        # iterate t = 1 .. n-1
        for t in range(1, n):
            lb = lower[t]
            lb_prev = lower[t - 1]
            dp2_curr = [NEG_INF] * n
            row = fruits[t]
            # For each feasible j at step t
            for j in range(lb, n):
                best = NEG_INF
                # from j-1
                pj = j - 1
                if pj >= lb_prev:
                    v = dp2_prev[pj]
                    if v > best:
                        best = v
                # from j
                if j >= lb_prev:
                    v = dp2_prev[j]
                    if v > best:
                        best = v
                # from j+1
                pj = j + 1
                if pj < n and pj >= lb_prev:
                    v = dp2_prev[pj]
                    if v > best:
                        best = v
                # if unreachable, skip
                if best < 0:
                    continue
                # reward at (t, j), skip if it's on the diagonal (j == t)
                add = 0 if j == t else row[j]
                dp2_curr[j] = best + add
            dp2_prev = dp2_curr

        child2_gain = dp2_prev[n - 1]  # at t = n-1, must end at j = n-1

        # DP for child3 (from (n-1, 0) to (n-1, n-1))
        # dp3_prev[i] = max fruits collected up to step t-1, ending at row i
        dp3_prev = [NEG_INF] * n
        # t = 0, child3 starts at i = n-1, column = 0
        i0 = lower[0]  # = n-1
        dp3_prev[i0] = fruits[i0][0] if i0 != 0 else 0

        for t in range(1, n):
            lb = lower[t]
            lb_prev = lower[t - 1]
            dp3_curr = [NEG_INF] * n
            # For each feasible i at step t
            for i in range(lb, n):
                best = NEG_INF
                # from i-1
                pi = i - 1
                if pi >= lb_prev:
                    v = dp3_prev[pi]
                    if v > best:
                        best = v
                # from i
                if i >= lb_prev:
                    v = dp3_prev[i]
                    if v > best:
                        best = v
                # from i+1
                pi = i + 1
                if pi < n and pi >= lb_prev:
                    v = dp3_prev[pi]
                    if v > best:
                        best = v
                # if unreachable, skip
                if best < 0:
                    continue
                # reward at (i, t), skip if it's on the diagonal (i == t)
                add = 0 if i == t else fruits[i][t]
                dp3_curr[i] = best + add
            dp3_prev = dp3_curr

        child3_gain = dp3_prev[n - 1]  # at t = n-1, must end at i = n-1

        # Total = diagonal collected by child1 + gains of child2 and child3 (excluding diagonal)
        return diag_sum + child2_gain + child3_gain