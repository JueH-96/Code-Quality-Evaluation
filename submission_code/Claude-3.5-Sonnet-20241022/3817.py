class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate total cells available on deck
        total_cells = n * n
        
        # Calculate maximum containers possible based on weight
        max_containers_by_weight = maxWeight // w
        
        # Return the minimum of total cells and max containers by weight
        return min(total_cells, max_containers_by_weight)