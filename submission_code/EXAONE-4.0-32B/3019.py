class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        fixed_left = moves.count('L')
        fixed_right = moves.count('R')
        underscores = len(moves) - fixed_left - fixed_right
        d1 = fixed_right - fixed_left - underscores
        d2 = fixed_right - fixed_left + underscores
        return max(abs(d1), abs(d2))