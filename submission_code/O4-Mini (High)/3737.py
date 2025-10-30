from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # We pair up house i with house n-1-i. Each pair (a,b) must have different colors,
        # and across consecutive pairs their left colors and right colors must differ.
        P = n // 2
        # All 6 possible (left_color, right_color) with left_color != right_color
        states = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        
        # Precompute, for each current state index ci, which previous state indices pi
        # satisfy: prev_left != cur_left and prev_right != cur_right
        transitions = []
        for (a, b) in states:
            allowed = []
            for pi, (x, y) in enumerate(states):
                if x != a and y != b:
                    allowed.append(pi)
            transitions.append(allowed)
        
        INF = 10**30
        # dp_prev[si] = minimum cost to paint pairs [0..k-1] ending in state si at pair k-1
        dp_prev = [INF] * 6
        # Initialize for the first pair k=0
        for si, (a, b) in enumerate(states):
            dp_prev[si] = cost[0][a] + cost[n-1][b]
        
        # Build DP for pairs k = 1 .. P-1
        for k in range(1, P):
            j = n - 1 - k
            dp_cur = [INF] * 6
            for ci, (a, b) in enumerate(states):
                base_cost = cost[k][a] + cost[j][b]
                best_prev = INF
                for pi in transitions[ci]:
                    if dp_prev[pi] < best_prev:
                        best_prev = dp_prev[pi]
                dp_cur[ci] = best_prev + base_cost
            dp_prev = dp_cur
        
        # Answer is the min cost among all ending states
        return min(dp_prev)