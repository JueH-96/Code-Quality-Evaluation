class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        blank_count = moves.count('_')
        
        # The furthest distance is achieved by moving all blanks in the same direction
        # as the majority of fixed moves
        return abs(left_count - right_count) + blank_count