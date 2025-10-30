class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate the total number of cells on the deck
        total_cells = n * n
        
        # Calculate the maximum number of containers that can be loaded without exceeding maxWeight
        max_possible_containers = maxWeight // w
        
        # The result is the minimum of total cells and the maximum possible containers
        return min(total_cells, max_possible_containers)