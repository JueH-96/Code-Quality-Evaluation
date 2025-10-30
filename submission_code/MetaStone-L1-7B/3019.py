class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_R = moves.count('R')
        count_L = moves.count('L')
        n_unders = moves.count('_')
        e0 = (count_R - count_L) - n_unders
        e_max = (count_R - count_L) + n_unders
        return max(abs(e0), abs(e_max))