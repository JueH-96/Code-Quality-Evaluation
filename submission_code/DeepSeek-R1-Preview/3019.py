class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_ = len(moves) - count_L - count_R
        f_max = (count_R - count_L) + count_
        f_min = (count_R - count_L) - count_
        return max(abs(f_max), abs(f_min))