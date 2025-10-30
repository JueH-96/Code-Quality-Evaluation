class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        blank_count = moves.count('_')

        # The maximum distance can be achieved by moving left as much as possible
        # and using blanks to move left
        max_left = left_count + blank_count

        # The maximum distance can be achieved by moving right as much as possible
        # and using blanks to move right
        max_right = right_count + blank_count

        # The furthest distance from the origin is the maximum of the two
        return max(max_left, max_right)