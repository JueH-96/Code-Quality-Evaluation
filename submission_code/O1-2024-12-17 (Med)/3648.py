class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        """
        We have a dungeon of size n x n, and three children starting at:
          Child #1: (0, 0)
          Child #2: (0, n-1)
          Child #3: (n-1, 0)
        Each child must make exactly n-1 moves to reach (n-1, n-1). Their
        move rules force them so that:
          - Child #1's path is fixed as (0,0)->(1,1)->...->(n-1,n-1).
            (Because each step must increment i by 1 or j by 1, and we only
             have n-1 steps to get both i and j from 0 to n-1, so it must
             move diagonally each time.)
          - Child #2 makes a path from (0,n-1) down to (n-1,n-1), but each
            step is (i+1, j+{ -1, 0, 1 }), staying within the grid.
            This means i goes from 0 to n-1 (one per step), while j can
            move left, right, or stay, as long as it ends at n-1 after n-1
            steps.
          - Child #3 makes a path from (n-1,0) to (n-1,n-1), each step is
            (i+{ -1, 0, 1 }, j+1 ), staying within the grid.
            So j goes from 0 to n-1, while i moves up, down, or stays, and
            ends at i = n-1.

        We want to collect as many fruits as possible, counting each cell's
        fruits at most once no matter how many children visit it.

        Key observation:
          Child #1's path is forced (the main diagonal cells (t, t)).
          For each step t (0 <= t < n), we then have:
            - child #1 at (t, t)
            - child #2 at (t, j2)
            - child #3 at (i3, t)

          Where j2 and i3 each evolve by +/-1 or 0 as t increases by 1,
          starting from j2(0)=n-1, i3(0)=n-1, and ending with j2(n-1)=n-1,
          i3(n-1)=n-1.  

        We can do a dynamic programming over t, j2, i3:
          DP(t, j2, i3) = maximum sum of unique fruits collected up to step t,
                          if at step t child #2 is at (t,j2), child #3 is
                          at (i3,t).

        Transition:
          DP(t, j2, i3) = stepValue(t, j2, i3) + max of
                          DP(t-1, j2_old, i3_old)
          where j2_old in [j2-1, j2, j2+1],
                i3_old in [i3-1, i3, i3+1],
                all within [0..n-1].

        stepValue(t, j2, i3) counts fruits of the three visited cells
        (t,t), (t,j2), (i3,t) exactly once each if distinct.

        Because n can be up to 1000, a naive DP is O(n * n^2 * 9) = O(9n^3) ~ 9e9,
        which is borderline in Python.  We implement it carefully and only compute
        over the "band" of feasible j2, i3 for each t (|j2 - (n-1)| <= t, etc.).
        Even so, it is quite large.  In practice a well-optimized solution can pass.

        Implementation outline:
          - We'll keep two 2D arrays dpPrev, dpCurr (size n x n) for DP states,
            using -inf for invalid ones.
          - Initialize dpPrev for t=0 with dpPrev[n-1][n-1] = stepValue(0, n-1, n-1).
          - For each t from 1 to n-1:
              * We compute dpCurr via the standard transitions from dpPrev.
              * We only loop j2 in the feasible range [n-1 - t .. n-1 + t],
                i3 in the same feasible range, clamped to [0..n-1].
              * For each (j2,i3), find the maximum dpPrev in its 3x3 neighborhood,
                then add stepValue(t, j2, i3).
            Move dpCurr -> dpPrev for the next iteration.
          - At the end, the DP value at dpPrev[n-1][n-1] is the maximum sum for
            all three children (because stepValue includes child #1's diagonal
            cell each time).

        This solves the problem with careful coding.

        """

        import sys
        sys.setrecursionlimit(10**7)
        n = len(fruits)
        # For convenience, define a small function to get the "step contribution"
        # from cells visited at time t by (child1=(t,t), child2=(t,j2), child3=(i3,t))
        def step_value(t, j2, i3):
            # Collect fruits from up to 3 distinct cells. If any overlap, count once.
            # cellA = (t, t)
            # cellB = (t, j2)
            # cellC = (i3, t)
            if j2 == t:
                if i3 == t:
                    # All three children in the same cell (t,t)
                    return fruits[t][t]
                else:
                    # Child #1 & #2 in (t,t), child #3 in (i3,t)
                    return fruits[t][t] + fruits[i3][t]
            else:
                if i3 == t:
                    # Child #1 & #3 in (t,t), child #2 in (t,j2)
                    return fruits[t][t] + fruits[t][j2]
                else:
                    # All distinct
                    return fruits[t][t] + fruits[t][j2] + fruits[i3][t]

        # We'll store DP states in 2D arrays: dp[r][c] = best sum if child#2 is at (t,r)
        # and child#3 is at (c,t).  We'll call r=j2, c=i3 to be consistent:
        INF_NEG = float('-inf')
        dpPrev = [[INF_NEG]*n for _ in range(n)]
        
        # Initialization: t=0 => child#2 at (0, n-1), child#3 at (n-1, 0)
        # dp(0, n-1, n-1) = step_value(0, n-1, n-1)
        dpPrev[n-1][n-1] = step_value(0, n-1, n-1)

        # We'll iterate t from 1..n-1
        for t in range(1, n):
            dpCurr = [[INF_NEG]*n for _ in range(n)]
            
            low = max(0, (n-1) - t)
            high = min(n-1, (n-1) + t)
            
            # To avoid repeated 3x3 scanning for each cell in [low..high]^2,
            # we do it naively but only over the band (2t+1 x 2t+1).
            # That's about O( (2t+1)^2 * 9 ) = O(9*(2t+1)^2 ) each iteration,
            # summing to O(n^3) overall.
            
            # Precompute a 3x3 max neighborhood for dpPrev in that band:
            # We'll call it bestAround[r][c].
            bestAround = [[INF_NEG]* (high+1) for _ in range(high+1)]
            # We'll store only for r,c in [low..high].
            # We'll fill it carefully just for that subregion.
            
            for r in range(low, high+1):
                for c in range(low, high+1):
                    mx = INF_NEG
                    for dr in (-1, 0, 1):
                        rr = r + dr
                        if 0 <= rr < n:
                            for dc in (-1, 0, 1):
                                cc = c + dc
                                if 0 <= cc < n:
                                    val = dpPrev[rr][cc]
                                    if val > mx:
                                        mx = val
                    bestAround[r][c] = mx
            
            # Now compute dpCurr within the band
            for j2 in range(low, high+1):
                for i3 in range(low, high+1):
                    base = bestAround[j2][i3]
                    if base == INF_NEG:
                        continue
                    dpCurr[j2][i3] = base + step_value(t, j2, i3)
            
            dpPrev = dpCurr
        
        # The final answer is dpPrev[n-1][n-1], which corresponds to
        # t=n-1, child#2 at (n-1, n-1), child#3 at (n-1, n-1).
        return dpPrev[n-1][n-1]