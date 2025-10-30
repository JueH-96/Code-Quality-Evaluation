class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The maximum length can be achieved by using all available strings
        # without creating the substrings "AAA" or "BBB".

        # We can use all 'z' strings "AB" as they do not create any issues.
        # For 'x' strings "AA" and 'y' strings "BB", we need to ensure we do not
        # have three consecutive 'A's or 'B's.

        # We can use min(x, y + 1) strings of "AA" and min(y, x + 1) strings of "BB"
        # without creating "AAA" or "BBB".

        # The reasoning is that we can always place an "AB" or "BB" between "AA"s
        # and an "AA" or "AB" between "BB"s to avoid three consecutive same characters.

        max_length = 0

        # Add all 'z' strings "AB"
        max_length += 2 * z

        # Add 'x' strings "AA"
        max_length += 2 * min(x, y + 1)

        # Add 'y' strings "BB"
        max_length += 2 * min(y, x + 1)

        return max_length