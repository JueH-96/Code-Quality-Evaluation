from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # We have n even. Pair up house i and house n-1-i for i in [0..n/2-1].
        # In each pair (rung) k we assign colors (c1,c2) with c1!=c2.
        # Adjacent constraints:
        #  - house k and k+1: c1_k != c1_{k+1}
        #  - house n-1-k and n-2-k: c2_k != c2_{k+1}
        # This forms a "ladder" of m = n/2 rungs. Each rung has 6 possible states.
        # We do DP over rungs.
        m = n // 2
        INF = 10**30
        
        # Enumerate all valid rung states: (c1,c2) with c1!=c2, colors in [0,1,2]
        states = []
        for c1 in range(3):
            for c2 in range(3):
                if c1 != c2:
                    states.append((c1, c2))
        # Precompute cost_sum[k][s] = cost at rung k with state s
        # where k corresponds to houses k and n-1-k
        cost_sum = [[0]*6 for _ in range(m)]
        for k in range(m):
            i = k
            j = n - 1 - k
            for si, (c1, c2) in enumerate(states):
                cost_sum[k][si] = cost[i][c1] + cost[j][c2]
        
        # Precompute valid transitions between states si -> sj
        # valid if c1_k != c1_{k+1} and c2_k != c2_{k+1}
        ok = [[False]*6 for _ in range(6)]
        for si, (a1, a2) in enumerate(states):
            for sj, (b1, b2) in enumerate(states):
                if a1 != b1 and a2 != b2:
                    ok[si][sj] = True
        
        # DP rolling arrays
        dp_prev = [INF]*6
        dp_curr = [INF]*6
        
        # Base case: rung 0
        for si in range(6):
            dp_prev[si] = cost_sum[0][si]
        
        # Transition
        for k in range(1, m):
            for sj in range(6):
                best = INF
                cs = cost_sum[k][sj]
                # try all previous states
                row = ok  # alias
                for si in range(6):
                    if row[si][sj]:
                        v = dp_prev[si] + cs
                        if v < best:
                            best = v
                dp_curr[sj] = best
            dp_prev, dp_curr = dp_curr, dp_prev
        
        # answer is min over last rung
        return min(dp_prev)