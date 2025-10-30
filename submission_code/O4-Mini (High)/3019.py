class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count fixed lefts, rights, and flexible moves ('_')
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_blank = moves.count('_')
        
        # The maximum distance is the existing imbalance plus all blanks
        return abs(count_R - count_L) + count_blank