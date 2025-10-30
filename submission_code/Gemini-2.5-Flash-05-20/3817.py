class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        """
        Calculates the maximum number of containers that can be loaded onto a ship.

        The number of containers is limited by two factors:
        1. The physical space available on the n x n cargo deck.
        2. The ship's maximum weight capacity.

        Args:
            n: The size of the square cargo deck (n x n).
            w: The weight of a single container.
            maxWeight: The maximum weight capacity of the ship.

        Returns:
            The maximum number of containers that can be loaded.
        """

        # Calculate the maximum number of containers based on the physical space (deck cells).
        # An n x n deck has n * n cells.
        max_containers_by_space = n * n

        # Calculate the maximum number of containers that can be loaded based on the
        # ship's weight capacity. Integer division is used because we can only load
        # whole containers.
        max_containers_by_weight = maxWeight // w

        # The actual maximum number of containers must satisfy both constraints.
        # Therefore, it is the minimum of the two calculated maximums.
        return min(max_containers_by_space, max_containers_by_weight)