class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_ = moves.count('_')
        
        # Option 1: Convert all '_' to 'L'
        pos_left = - (count_L + count_) + count_R
        
        # Option 2: Convert all '_' to 'R'
        pos_right = (count_R + count_) - count_L
        
        return max(abs(pos_left), abs(pos_right))