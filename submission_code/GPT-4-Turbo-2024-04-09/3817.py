class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        total_cells = n * n
        max_possible_containers = maxWeight // w
        return min(total_cells, max_possible_containers)