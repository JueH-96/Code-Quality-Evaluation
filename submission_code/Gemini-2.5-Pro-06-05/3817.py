class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        """
        Calculates the maximum number of containers that can be loaded onto an n x n cargo deck.

        Args:
            n: The side length of the square cargo deck.
            w: The weight of a single container.
            maxWeight: The maximum total weight the ship can carry.

        Returns:
            The maximum number of containers that can be loaded.
        """

        # 1. Calculate the maximum number of containers limited by deck space.
        # The deck is an n x n grid, so it has n * n cells.
        # Each cell can hold one container.
        max_containers_by_space = n * n

        # 2. Calculate the maximum number of containers limited by weight capacity.
        # The total weight (number of containers * w) must not exceed maxWeight.
        # So, num_containers * w <= maxWeight  =>  num_containers <= maxWeight / w.
        # We use integer division since we can only load whole containers.
        max_containers_by_weight = maxWeight // w

        # The actual number of containers we can load is the minimum of the two limits.
        # We are constrained by whichever is the smaller value: the available space
        # or the weight capacity.
        return min(max_containers_by_space, max_containers_by_weight)