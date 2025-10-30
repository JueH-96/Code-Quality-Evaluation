class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        max_possible = n * n
        total_weight = max_possible * w
        if total_weight <= maxWeight:
            return max_possible
        else:
            return min(max_possible, maxWeight // w)