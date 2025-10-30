class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # If x > y, we can use all y "BB"s, all z "AB"s, and y+1 "AA"s
        # If x <= y, we can use all x "AA"s, all z "AB"s, and x "BB"s
        if x > y:
            return 2 * (2 * y + z + 1)
        else:
            return 2 * (x + y + z)