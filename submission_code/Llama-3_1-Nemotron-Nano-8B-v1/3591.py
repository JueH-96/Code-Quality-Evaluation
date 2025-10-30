from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute next_sum and prev_sum tables
        next_sum = [[0] * 26 for _ in range(26)]
        prev_sum = [[0] * 26 for _ in range(26)]
        
        # Populate next_sum
        for a in range(26):
            current_next = 0
            for step in range(1, 26):
                current_char = (a + step - 1) % 26
                current_next += nextCost[current_char]
                next_sum[a][step] = current_next
        
        # Populate prev_sum
        for a in range(26):
            current_prev = 0
            for step in range(1, 26):
                current_char = (a - step + 1) % 26
                current_prev += previousCost[current_char]
                prev_sum[a][step] = current_prev
        
        total_cost = 0
        for sc, tc in zip(s, t):
            a = ord(sc) - ord('a')
            b = ord(tc) - ord('a')
            
            # Calculate cost for next direction
            steps_next = (b - a) % 26
            cost_next = next_sum[a][steps_next]
            
            # Calculate cost for previous direction
            steps_prev = (a - b) % 26
            cost_prev = prev_sum[a][steps_prev]
            
            total_cost += min(cost_next, cost_prev)
        
        return total_cost