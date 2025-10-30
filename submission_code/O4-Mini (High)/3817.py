class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Total slots on the deck
        total_slots = n * n
        # Maximum containers by weight constraint
        max_by_weight = maxWeight // w
        # We cannot exceed the number of slots
        return min(total_slots, max_by_weight)