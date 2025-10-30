from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute forward and backward cost tables
        forward_cost = [[0] * 26 for _ in range(26)]
        for c in range(26):
            current_sum = 0
            forward_cost[c][0] = 0  # 0 steps
            for k in range(1, 26):
                pos = (c + k - 1) % 26
                current_sum += nextCost[pos]
                forward_cost[c][k] = current_sum
        
        backward_cost = [[0] * 26 for _ in range(26)]
        for c in range(26):
            current_sum = 0
            backward_cost[c][0] = 0  # 0 steps
            for k in range(1, 26):
                pos = (c - (k - 1)) % 26
                current_sum += previousCost[pos]
                backward_cost[c][k] = current_sum
        
        total = 0
        for sc, tc in zip(s, t):
            c = ord(sc) - ord('a')
            d = ord(tc) - ord('a')
            if c == d:
                continue
            k_forward = (d - c) % 26
            k_backward = (c - d) % 26
            cost_forward = forward_cost[c][k_forward]
            cost_backward = backward_cost[c][k_backward]
            total += min(cost_forward, cost_backward)
        
        return total