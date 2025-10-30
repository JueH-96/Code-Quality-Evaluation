class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_ = moves.count('_')
        
        # The strategy is to add all '_' moves in whichever direction
        # gives the largest absolute displacement (left or right).
        return abs(count_R - count_L) + count_