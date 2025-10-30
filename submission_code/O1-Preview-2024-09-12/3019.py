class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        numL = moves.count('L')
        numR = moves.count('R')
        num_ = moves.count('_')
        D1 = (numR + num_) - numL
        D2 = numR - (numL + num_)
        return max(abs(D1), abs(D2))