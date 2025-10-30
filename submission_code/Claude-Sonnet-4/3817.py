class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers that can fit on the deck
        deck_capacity = n * n
        
        # Maximum containers allowed by weight constraint
        weight_capacity = maxWeight // w
        
        # Return the minimum of both constraints
        return min(deck_capacity, weight_capacity)