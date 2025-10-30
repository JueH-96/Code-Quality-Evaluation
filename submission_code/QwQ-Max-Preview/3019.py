class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_ = len(moves) - count_L - count_R
        initial_diff = count_R - count_L
        option1 = initial_diff + count_
        option2 = initial_diff - count_
        return max(abs(option1), abs(option2))