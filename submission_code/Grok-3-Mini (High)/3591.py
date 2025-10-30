import math
from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Compute cumulative sums for nextCost and previousCost
        cum_next = [0] * 27
        for i in range(26):
            cum_next[i + 1] = cum_next[i] + nextCost[i]
        
        cum_prev = [0] * 27
        for i in range(26):
            cum_prev[i + 1] = cum_prev[i] + previousCost[i]
        
        # Define a helper function to compute circular sum
        def circular_sum(cum_sum, start, D):
            if D == 0:
                return 0
            if start + D <= 26:
                return cum_sum[start + D] - cum_sum[start]
            else:
                sum_part1 = cum_sum[26] - cum_sum[start]
                E = (start + D - 1) % 26
                sum_part2 = cum_sum[E + 1]
                return sum_part1 + sum_part2
        
        # Initialize total cost
        total_cost = 0
        n = len(s)
        
        # Iterate over each character in s and t
        for i in range(n):
            start_idx = ord(s[i]) - ord('a')
            end_idx = ord(t[i]) - ord('a')
            
            # Calculate delta for clockwise and counterclockwise
            delta_cw = (end_idx - start_idx) % 26
            delta_ccw = (start_idx - end_idx) % 26
            
            # Calculate cost for clockwise path
            cost_cw = circular_sum(cum_next, start_idx, delta_cw)
            
            # Calculate start index for counterclockwise path
            start_ccw_idx = (start_idx - delta_ccw + 1) % 26
            # Calculate cost for counterclockwise path
            cost_ccw = circular_sum(cum_prev, start_ccw_idx, delta_ccw)
            
            # Take the minimum cost and add to total
            min_cost = min(cost_cw, cost_ccw)
            total_cost += min_cost
        
        return total_cost