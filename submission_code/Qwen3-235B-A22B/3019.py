class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        underscores = moves.count('_')
        return abs(count_R - count_L) + underscores