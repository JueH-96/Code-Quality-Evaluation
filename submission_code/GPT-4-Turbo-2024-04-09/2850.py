class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Sort the counts of "AA" and "BB" so that we always have a <= b
        # This simplifies the logic to always try to balance the larger count with the smaller
        a, b = sorted([x, y])
        
        # If we have more "AB" than we can sandwich between "AA" and "BB", we use as many as possible
        if z >= b:
            # We can use all b and a, and z can be used as much as possible
            # Each "AB" can be placed between "AA" and "BB" or vice versa, consuming one "AA" or "BB"
            # After placing b "AB"s, we have:
            # - b "AB" pairs fully placed
            # - Remaining z - b "AB"s can only be placed if there are slots available
            # The remaining "AB"s can be placed at the start or end or between any "AA" or "BB"
            # We can place at most one more "AB" than the number of remaining "AA" or "BB"
            return 2 * (a + b + min(z, b + 1))
        else:
            # If z < b, we can use all z, and we need to balance the remaining "AA" and "BB"
            # After using z "AB"s, we have:
            # - a - z "AA" left
            # - b - z "BB" left
            # We can place at most one more "AA" or "BB" than the other
            remaining_a = a - z
            remaining_b = b - z
            # We can place all of the smaller count and at most one more of the larger count
            return 2 * (z + min(remaining_a, remaining_b) + min(remaining_a, remaining_b + 1))