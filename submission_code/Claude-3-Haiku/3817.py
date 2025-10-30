class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate the maximum number of containers that can be loaded
        max_containers = min(n * n, maxWeight // w)
        return max_containers