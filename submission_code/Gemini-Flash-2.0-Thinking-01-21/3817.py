class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Calculate the total number of available cells on the deck.
        # This is the physical limit on the number of containers.
        physical_limit = n * n

        # Calculate the maximum number of containers that can be loaded
        # without exceeding the ship's maximum weight capacity.
        # Since each container weighs 'w', if we load 'k' containers, the total weight is 'k * w'.
        # We need k * w <= maxWeight.
        # So, k <= maxWeight / w.
        # Since we can only load whole containers, the maximum number of containers
        # allowed by weight is maxWeight // w (integer division).
        # If w > maxWeight, this correctly evaluates to 0.
        # We must handle the case where w is 0, though the constraints say w >= 1.
        # Given w >= 1, maxWeight // w is safe.
        weight_limit = maxWeight // w

        # The actual maximum number of containers is the minimum of the physical limit
        # and the weight limit, as both constraints must be satisfied.
        return min(physical_limit, weight_limit)