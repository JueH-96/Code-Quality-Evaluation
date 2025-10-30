class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Total cells available on the deck
        total_cells = n * n
        
        # Maximum containers based on weight constraint
        max_containers_by_weight = maxWeight // w
        
        # Return the minimum of the two constraints
        return min(total_cells, max_containers_by_weight)