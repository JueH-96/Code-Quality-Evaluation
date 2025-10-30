from typing import List

class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        """
        DP   dp[i][j][k] – maximum profit reaching cell (i,j) having already
        used exactly k neutralisations (k = 0,1,2).
        Move only right or down, so each cell can be reached from the top or
        from the left.  For a negative cell we may either
              • pay the loss          (keep k)
              • neutralise the robber (k -> k+1, k+1 ≤ 2)
        For a non-negative cell we must pay/gain the value (keep k).

        We keep only the previous row to obtain O(m·n·3) time
        and O(n·3) memory.
        """
        if not coins or not coins[0]:
            return 0

        m, n = len(coins), len(coins[0])
        K = 2                              # maximum neutralisations allowed
        NEG = -10**18                      # value lower than any possible sum

        # prev[j][k] – previous row DP for column j and k neutralisations
        prev = [[NEG]*(K+1) for _ in range(n)]

        for i in range(m):
            # current row we are filling
            curr = [[NEG]*(K+1) for _ in range(n)]
            for j in range(n):
                val = coins[i][j]

                # container for the best results for this cell
                best = [NEG]*(K+1)

                if i == 0 and j == 0:          # starting cell, handle separately
                    if val >= 0:
                        best[0] = val                  # no neutralisation
                    else:                             # negative start cell
                        best[0] = val                 # pay the loss
                        best[1] = 0                   # or neutralise it
                else:
                    # collect possible predecessors (top, left)
                    sources = []
                    if i > 0:
                        sources.append(prev[j])
                    if j > 0:
                        sources.append(curr[j-1])

                    for src in sources:               # examine each predecessor
                        for k_prev, prev_val in enumerate(src):
                            if prev_val == NEG:       # unreachable state
                                continue
                            if val >= 0:
                                # must take (gain) the positive coins
                                best[k_prev] = max(best[k_prev],
                                                   prev_val + val)
                            else:
                                # pay the loss (no neutralisation here)
                                best[k_prev] = max(best[k_prev],
                                                   prev_val + val)
                                # or neutralise if we still can
                                if k_prev < K:
                                    best[k_prev+1] = max(best[k_prev+1],
                                                         prev_val)

                curr[j] = best                       # store results for cell

            prev = curr                              # next row becomes previous

        return max(prev[-1])                         # best over 0,1,2 neutralisations