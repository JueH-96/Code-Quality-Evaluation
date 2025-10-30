from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        """
        We need to pick a subset S of walls to be painted by the paid painter.
        Let v_i = time[i] + 1.  If we pick walls in S, the total "capacity" is
        sum(v_i) = sum(time[i]) + |S|.  The free painter can paint at most
        sum(time[i]) walls (1 per time unit) while the paid painter is busy,
        so the total number of walls painted is |S| + sum(time[i]).  We need
        that to be >= n, i.e. sum(v_i) >= n.  We want to minimize the sum of
        cost[i] over S.
        
        This becomes a 0/1 knapsack: items of weight v_i, cost c_i, target
        capacity n, minimize total cost to reach at least weight n.  We can
        cap the DP states at n, since any sum >= n can be recorded in dp[n].
        dp[w] = minimum cost to reach total weight exactly w (or >=n if w=n).
        """
        n = len(cost)
        INF = 10**30
        # dp[w] = min cost to reach total "weight" w, 0 <= w <= n;
        # by convention dp[n] stands for ">= n"
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for c, t in zip(cost, time):
            v = t + 1
            # 0/1 knapsack: traverse backwards to avoid reuse of the same item
            for w in range(n, -1, -1):
                # new total weight
                nw = w + v
                if nw >= n:
                    nw = n
                # try taking this wall with the paid painter
                new_cost = dp[w] + c
                if new_cost < dp[nw]:
                    dp[nw] = new_cost
        
        return dp[n]