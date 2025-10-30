from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        half = n // 2
        # There are 3 colors: 0, 1, 2.
        # For each pair p (0 <= p < half), let left house index = p and right house = n-1-p.
        # We assign a color pair (c, d) with the constraint that c != d (mirror condition).
        # Additionally, from the overall linear order:
        # • Adjacent left houses must have different colors.
        # • Adjacent right houses (in reversed order) must have different colors.
        # Thus, for consecutive pairs p-1 and p, we require:
        #   left[p-1] != left[p] and right[p-1] != right[p].
        #
        # There are only 6 allowed color pair choices for each pair.
        # We'll use dynamic programming over pairs.
        
        # Initialize dp for pair 0: dictionary mapping (c, d) to cost.
        dp_prev = {}
        for c in range(3):
            for d in range(3):
                if c != d:
                    dp_prev[(c, d)] = cost[0][c] + cost[n-1][d]
        
        # Process pairs from 1 to half-1
        for p in range(1, half):
            dp_curr = {}
            left_index = p
            right_index = n - 1 - p
            for curr_c in range(3):
                for curr_d in range(3):
                    if curr_c == curr_d:
                        continue  # mirror constraint for current pair
                    # Cost for current pair assignment:
                    curr_cost = cost[left_index][curr_c] + cost[right_index][curr_d]
                    best_prev = float('inf')
                    # Transition from previous pair such that:
                    # previous left color != curr_c and previous right color != curr_d.
                    for (prev_c, prev_d), prev_cost in dp_prev.items():
                        if prev_c != curr_c and prev_d != curr_d:
                            if prev_cost < best_prev:
                                best_prev = prev_cost
                    if best_prev != float('inf'):
                        dp_curr[(curr_c, curr_d)] = curr_cost + best_prev
            dp_prev = dp_curr  # move to next pair
        
        # answer: the minimal cost among all assignments in the last pair
        return min(dp_prev.values()) if dp_prev else -1