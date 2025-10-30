class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate the total number of cells available on the deck.
        # Each cell can hold one container.
        # So, the maximum number of containers based on space is n*n.
        num_available_cells = n * n

        # Calculate the maximum number of containers that can be loaded
        # based on the weight constraint.
        # Each container weighs w. Total weight must not exceed maxWeight.
        # So, num_containers * w <= maxWeight
        # which means num_containers <= maxWeight / w.
        # Since w >= 1 (from constraints: 1 <= w <= 1000), division by zero is not a concern.
        # We need an integer number of containers, so use integer division.
        # If maxWeight is less than w, this will correctly result in 0.
        num_containers_by_weight_limit = maxWeight // w

        # The actual maximum number of containers is limited by both
        # the available space and the weight capacity.
        # Therefore, it's the minimum of these two values.
        max_num_containers = min(num_available_cells, num_containers_by_weight_limit)

        return max_num_containers