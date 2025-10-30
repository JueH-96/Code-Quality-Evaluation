class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_moves = moves.count('L') + moves.count('_')
        right_moves = moves.count('R') + moves.count('_')
        return abs(left_moves - right_moves)