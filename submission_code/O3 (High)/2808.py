from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        """
        We choose which walls will be painted by the paid painter.
        If we pay for wall i we:
          • spend  cost[i]   money
          • occupy the paid painter for time[i] units of time
            → during those time[i] units the free painter can paint
              at most time[i] other walls.
        So, paying for wall i lets us finish     1 + time[i]     walls in total
        (the wall itself plus up to time[i] free ones).

        The task is therefore:
            pick a subset of indices P
            such that      Σ(1 + time[i])  (i ∈ P) ≥ n
            and            Σ cost[i]       (i ∈ P) is minimum.
        This is a 0/1-knapsack where “value” = coverage (1+time[i])
        and “weight” = cost[i]; we need at least n “value”.

        n ≤ 500, so an O(n²) DP is fine.
        """
        n = len(cost)
        INF = 10**18

        # dp[k] = minimum money needed to be able to finish k walls
        # Only the first n entries are necessary (k never exceeds n).
        dp = [INF] * (n + 1)
        dp[0] = 0

        for c, t in zip(cost, time):
            cover = t + 1                    # walls finished if we pay for this one
            # iterate backwards to keep 0/1 semantics
            for k in range(n, -1, -1):
                if dp[k] == INF:
                    continue
                nk = min(n, k + cover)       # cap at n: “at least n” is enough
                dp[nk] = min(dp[nk], dp[k] + c)

        return dp[n]