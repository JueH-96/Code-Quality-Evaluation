from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # Precompute the sum for child1's fixed path (diagonal cells)
        fixed_sum = sum(fruits[i][i] for i in range(n))
        
        # We will use dp for child2 and child3 extra fruits.
        # dp[c2][r3] : maximum extra fruits collected so far when at step t,
        # with child2 at (t, c2) and child3 at (r3, t).
        NEG_INF = -10**9  # a sufficiently small value
        
        # initialize dp for t = 0. At t=0:
        # child1 at (0,0); child2 at (0, n-1); child3 at (n-1, 0)
        # rewards: for child2: (0, n-1) (if n-1 != 0) and for child3: (n-1,0) (if n-1 != 0)
        dp = [[NEG_INF] * n for _ in range(n)]
        # reward function for a given t and cell positions for child2 and child3
        def reward(t, c2, r3):
            # children 2 and 3 are entering cells (t, c2) and (r3, t) respectively.
            # But if c2==t or r3==t, that room is on child1's diagonal
            val1 = fruits[t][c2] if c2 != t else 0
            val2 = fruits[r3][t] if r3 != t else 0
            # if the two children land in the same room (and that cell is off diagonal), count once.
            if c2 == r3:
                # if c2==t then the cell is on the fixed diagonal so reward 0.
                return fruits[t][c2] if c2 != t else 0
            return val1 + val2
        
        # starting state at t = 0:
        start_val = reward(0, n - 1, n - 1)
        dp[n - 1][n - 1] = start_val
        
        # For t from 0 to n-2 (we already set the dp for t=0)
        # At each step, child2's row becomes t+1 and child3's column becomes t+1.
        for t in range(0, n - 1):
            # For a given t, the valid indices for the free coordinates:
            # For child2: c2 can only be in [max(0, n–1 - t), n–1] 
            # For child3: r3 can only be in [max(0, n–1 - t), n–1]
            new_dp = [[NEG_INF] * n for _ in range(n)]
            low = max(0, n - 1 - t)
            # iterate only over states that were reached at step t:
            for c2 in range(low, n):
                for r3 in range(low, n):
                    if dp[c2][r3] == NEG_INF:
                        continue
                    # now child2 is at (t, c2) and child3 is at (r3, t)
                    # they will make one move.
                    for dc in (-1, 0, 1):
                        nc2 = c2 + dc  # new column for child2; its row becomes t+1 automatically.
                        if not (0 <= nc2 < n):
                            continue
                        for dr in (-1, 0, 1):
                            nr3 = r3 + dr  # new row for child3; its column becomes t+1.
                            if not (0 <= nr3 < n):
                                continue
                            # add reward from cells at time t+1:
                            cand = dp[c2][r3] + reward(t + 1, nc2, nr3)
                            if cand > new_dp[nc2][nr3]:
                                new_dp[nc2][nr3] = cand
            dp = new_dp
        
        # At the end t = n-1, the destinations must be:
        # child2 at (n-1, n-1) i.e. c2 == n-1, and
        # child3 at (n-1, n-1) i.e. r3 == n-1.
        extra = dp[n - 1][n - 1]
        return fixed_sum + extra