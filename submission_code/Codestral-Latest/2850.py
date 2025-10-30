class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # If we have more 'AA' strings than 'BB' strings, we can swap them
        if x > y:
            x, y = y, x

        # Calculate the maximum number of 'AA' and 'BB' pairs we can use
        max_pairs = min(x, y)

        # The length contributed by 'AA' and 'BB' pairs
        length_from_pairs = 4 * max_pairs

        # The remaining 'AA' and 'BB' strings
        remaining_x = x - max_pairs
        remaining_y = y - max_pairs

        # The length contributed by the remaining 'AA' and 'BB' strings
        length_from_remaining = 2 * (remaining_x + remaining_y)

        # The length contributed by 'AB' strings
        length_from_ab = 2 * z

        # The total length of the new string
        total_length = length_from_pairs + length_from_remaining + length_from_ab

        return total_length