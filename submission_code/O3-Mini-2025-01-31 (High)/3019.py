class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the forced moves to the left and right
        left_count = moves.count('L')
        right_count = moves.count('R')
        # Count the underscores which represent moves we can choose either direction
        underscore_count = moves.count('_')
        
        # The net forced displacement is:
        #   right_count - left_count (which can be negative)
        # To maximize the absolute distance, we can assign all underscores to the side
        # that makes the total displacement further from 0.
        #
        # Thus, the furthest distance = abs(right_count - left_count) + underscore_count.
        return abs(right_count - left_count) + underscore_count