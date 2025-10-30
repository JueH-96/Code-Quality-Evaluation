class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The maximum length of the new string is the sum of the lengths of the strings
        # multiplied by their respective counts, plus the minimum of x and y multiplied by 2
        # because we can use "AA" and "BB" to form "AAAA" and "BBBB"
        return (x + y + z) * 2 + min(x, y) * 2