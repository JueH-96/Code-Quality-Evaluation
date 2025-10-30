import math

class Solution:
    """
    Calculates the maximum number of containers that can be loaded onto an n x n cargo deck.

    Args:
        n: The dimension of the square cargo deck (n x n).
        w: The weight of each container.
        maxWeight: The maximum total weight capacity of the ship.

    Returns:
        The maximum number of containers that can be loaded.
    """
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        """
        Calculates the maximum number of containers based on deck space and weight limit.

        The number of containers is limited by two factors:
        1. The total number of cells available on the n x n deck, which is n * n.
           This represents the maximum number of containers limited by physical space.
        2. The maximum number of containers whose total weight does not exceed maxWeight.
           If k is the number of containers, then k * w <= maxWeight.
           Thus, k <= maxWeight / w. The maximum integer k is floor(maxWeight / w).
           This represents the maximum number of containers limited by weight capacity.

        The function returns the minimum of these two limits, as both constraints must be satisfied.
        """

        # Calculate the total number of cells (maximum containers limited by space)
        total_cells = n * n

        # Calculate the maximum number of containers allowed by the weight limit.
        # We use integer division `//` which computes floor(maxWeight / w)
        # This is valid because the constraints state w >= 1, so no division by zero.
        max_containers_by_weight = maxWeight // w

        # The actual maximum number of containers we can load is the minimum
        # of the number of available cells and the number allowed by the weight limit.
        result = min(total_cells, max_containers_by_weight)

        return result