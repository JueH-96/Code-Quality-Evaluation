class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers based on available space
        max_by_space = n * n
        
        # Maximum containers based on weight constraint
        max_by_weight = maxWeight // w
        
        # Return the minimum of both constraints
        return min(max_by_space, max_by_weight)