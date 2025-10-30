class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate the total number of containers that can fit on the deck
        totalContainers = n * n
        
        # Calculate the maximum number of containers that can be loaded without exceeding maxWeight
        maxLoadableContainers = maxWeight // w
        
        # The result is the minimum of total containers and max loadable containers
        return min(totalContainers, maxLoadableContainers)