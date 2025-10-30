class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        """
        Binary–search the answer L (= maximal run length after the flips).
        For a fixed L   ⇒   can we turn the original string into ANY binary
        string whose runs are all ≤ L using ≤ numOps flips?
        
        For that feasibility check we compute the minimum number of flips
        required with dynamic programming.

        dp[i][c][ℓ] – minimal flips for the prefix ending at position i
                      if the current character is c (0/1) and the length of
                      the current run equals ℓ   (1 … L).

        Transition (from position i-1 to i):
             – keep the same character (run grows by 1, must stay ≤ L)
             – switch character            (run length becomes 1)

        Complexity
            DP for a fixed L   :  O(n · L · 2)   ( ≤ 2·10⁶ for n = 1000 )
            Binary search log n:  O(log n)
            So total            :  O(n · L · log n)  ≤ 1 – 2·10⁷ steps,
            absolutely fine for the constraints.
        """
        n = len(s)
        a = [0 if ch == '0' else 1 for ch in s]          # 0/1 list
        
        # ------------------------------------------------------------------
        def feasible(L: int) -> bool:
            INF = numOps + 1                              # > maximum allowed
            # dp_prev[char][runLen]  (runLen : 1 … L)
            dp_prev = [[INF] * (L + 1) for _ in range(2)]
            # position 0 initialisation
            dp_prev[0][1] = a[0] ^ 0                      # cost flip to 0 ?
            dp_prev[1][1] = a[0] ^ 1                      # cost flip to 1 ?
            
            for i in range(1, n):
                dp_cur = [[INF] * (L + 1) for _ in range(2)]
                for c_prev in (0, 1):
                    row_prev = dp_prev[c_prev]
                    for len_prev in range(1, L + 1):
                        prev_cost = row_prev[len_prev]
                        if prev_cost > numOps:            # pruning
                            continue
                        # 1) keep same character
                        if len_prev < L:
                            cost_same = prev_cost + (a[i] ^ c_prev)
                            if cost_same < dp_cur[c_prev][len_prev + 1]:
                                dp_cur[c_prev][len_prev + 1] = cost_same
                        # 2) switch character
                        c_new = 1 - c_prev
                        cost_switch = prev_cost + (a[i] ^ c_new)
                        if cost_switch < dp_cur[c_new][1]:
                            dp_cur[c_new][1] = cost_switch
                dp_prev = dp_cur
            
            # minimal flips among all end states
            min_cost = min(min(dp_prev[c][1:]) for c in (0, 1))
            return min_cost <= numOps
        # ------------------------------------------------------------------
        
        # Binary search for the minimal L
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo