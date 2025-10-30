class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        """
        Calculates the maximum number of containers that can be loaded onto an n x n deck
        without exceeding the maximum weight capacity.

        Args:
            n: The size of the square deck (n x n).
            w: The weight of each container.
            maxWeight: The maximum total weight the ship can hold.

        Returns:
            The maximum number of containers that can be loaded.
        """
        # The total number of cells (maximum possible containers based on space)
        max_physical_containers = n * n

        # The maximum number of containers based on weight capacity
        # We use integer division // to get the maximum integer number of containers
        # that fit within the maxWeight, since maxWeight, w are integers and w >= 1.
        max_weight_capacity_containers = maxWeight // w

        # The actual maximum number of containers is limited by the smaller of
        # the physical space and the weight capacity.
        return min(max_physical_containers, max_weight_capacity_containers)