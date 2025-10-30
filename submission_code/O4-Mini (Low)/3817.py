class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Total available slots on the deck
        total_slots = n * n
        # Maximum number of containers we can load without exceeding maxWeight
        max_by_weight = maxWeight // w
        # The result is the minimum of available slots and weight-based limit
        return min(total_slots, max_by_weight)