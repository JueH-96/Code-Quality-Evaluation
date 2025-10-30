class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        underscore_count = moves.count('_')

        # To maximize the distance, we should use all '_' in the direction that has fewer moves
        if left_count < right_count:
            left_count += underscore_count
        else:
            right_count += underscore_count

        # The furthest distance from the origin is the absolute difference between left and right moves
        return abs(left_count - right_count)