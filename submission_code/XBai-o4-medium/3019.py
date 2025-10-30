class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_R = moves.count('R')
        count_L = moves.count('L')
        x = moves.count('_')
        D = count_R - count_L
        return max(abs(D + x), abs(D - x))