class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We can use all z "AB" strings without any restrictions.
        # For "AA" and "BB" strings, we can interleave them to avoid "AAA" or "BBB".
        # The maximum number of pairs we can form is min(x, y), and each pair contributes 4 characters ("AABB" or "BBAA").
        # If x != y, we can add one more string of the type that has one more count.
        # Each "AA" or "BB" string contributes 2 characters.
        return 2 * (2 * min(x, y) + (x != y) + z)