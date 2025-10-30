class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers possible by deck capacity
        deck_capacity = n * n
        # Maximum containers possible by weight limit
        weight_limit_containers = maxWeight // w
        # The answer is the minimum of these two values
        return min(deck_capacity, weight_limit_containers)