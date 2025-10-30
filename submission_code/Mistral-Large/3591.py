from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        def char_to_index(c):
            return ord(c) - ord('a')

        total_cost = 0
        for i in range(len(s)):
            if s[i] != t[i]:
                idx_s = char_to_index(s[i])
                idx_t = char_to_index(t[i])

                # Calculate the cost to shift s[i] to t[i] in both directions
                forward_cost = 0
                backward_cost = 0

                if idx_s < idx_t:
                    forward_cost = sum(nextCost[idx_s:idx_t])
                    backward_cost = sum(previousCost[idx_t+1:]) + sum(previousCost[:idx_s+1])
                else:
                    forward_cost = sum(nextCost[idx_s+1:]) + sum(nextCost[:idx_t+1])
                    backward_cost = sum(previousCost[idx_t:idx_s])

                # Choose the minimum cost
                total_cost += min(forward_cost, backward_cost)

        return total_cost