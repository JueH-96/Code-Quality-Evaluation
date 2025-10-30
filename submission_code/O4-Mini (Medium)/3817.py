class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers by deck capacity
        max_by_cells = n * n
        # Maximum containers by weight constraint
        max_by_weight = maxWeight // w
        # The answer is the smaller of the two limits
        return min(max_by_cells, max_by_weight)