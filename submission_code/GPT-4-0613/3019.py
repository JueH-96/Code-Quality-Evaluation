class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_moves = moves.count('L')
        right_moves = moves.count('R')
        blank_moves = moves.count('_')
        return max(left_moves, right_moves) + blank_moves