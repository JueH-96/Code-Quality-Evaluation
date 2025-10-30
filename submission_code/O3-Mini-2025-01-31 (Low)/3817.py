class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Total number of container slots in the n x n deck.
        total_slots = n * n
        
        # Maximum containers allowed by the ship's weight capacity.
        max_by_weight = maxWeight // w
        
        # Return the number of containers that can actually be loaded.
        return min(total_slots, max_by_weight)