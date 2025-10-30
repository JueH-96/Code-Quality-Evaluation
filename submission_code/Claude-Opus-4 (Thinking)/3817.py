class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers based on weight limit
        max_by_weight = maxWeight // w
        
        # Maximum containers based on deck capacity
        max_by_deck = n * n
        
        # Return the minimum of the two constraints
        return min(max_by_weight, max_by_deck)