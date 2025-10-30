from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        total_cost = 0
        # Define helper functions to compute the cost from one letter to target letter using forward/backward operations.
        def forward_cost(start: int, target: int) -> int:
            # if start == target, cost is 0 (zero moves)
            if start == target:
                return 0
            cost = 0
            cur = start
            # moving forward in the alphabet, wrap around using modulo 26.
            while cur != target:
                # cost incurred depends on the current letter's next cost.
                cost += nextCost[cur]
                cur = (cur + 1) % 26
            return cost

        def backward_cost(start: int, target: int) -> int:
            if start == target:
                return 0
            cost = 0
            cur = start
            while cur != target:
                # cost incurred depends on the current letter's previous cost.
                cost += previousCost[cur]
                cur = (cur - 1) % 26
            return cost

        # For each character, pick the minimal cost direction.
        for cs, ct in zip(s, t):
            start = ord(cs) - ord('a')
            target = ord(ct) - ord('a')
            # Calculate cost for both directions
            cost_forward = forward_cost(start, target)
            cost_backward = backward_cost(start, target)
            total_cost += min(cost_forward, cost_backward)
        return total_cost