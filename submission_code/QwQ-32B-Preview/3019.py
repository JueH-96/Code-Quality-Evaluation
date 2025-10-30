class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        fixed_R = moves.count('R')
        fixed_L = moves.count('L')
        wild = moves.count('_')
        pos1 = fixed_R - fixed_L - wild
        pos2 = fixed_R - fixed_L + wild
        return max(abs(pos1), abs(pos2))