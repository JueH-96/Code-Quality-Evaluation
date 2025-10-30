class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        """
        Calculates the maximum number of containers that can be loaded onto
        an n x n deck given the weight of each container and the ship's
        maximum weight capacity.

        Args:
            n: The size of the square deck (n x n).
            w: The weight of a single container.
            maxWeight: The maximum total weight the ship can carry.

        Returns:
            The maximum number of containers that can be loaded.
        """
        # Calculate the total number of cells (potential container spots) on the deck.
        total_cells = n * n

        # Calculate the maximum number of containers that can be loaded
        # based solely on the ship's weight capacity.
        # Each container weighs 'w', so the maximum number is maxWeight / w.
        # Since we can only load whole containers, we use integer division (floor).
        # If w is 0 or maxWeight is less than w, this correctly results in 0 containers
        # (though constraints say w >= 1).
        max_by_weight = maxWeight // w

        # The actual maximum number of containers is limited by whichever constraint
        # is more restrictive: the physical space on the deck (total_cells) or
        # the ship's weight capacity (max_by_weight).
        # We take the minimum of these two values.
        return min(total_cells, max_by_weight)