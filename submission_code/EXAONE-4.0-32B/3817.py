class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        total_cells = n * n
        containers_by_weight = maxWeight // w
        return min(total_cells, containers_by_weight)