class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the number of 'L', 'R', and '_' in the moves string
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_blank = moves.count('_')
        
        # The furthest distance is the difference between the counts of 'L' and 'R'
        # plus the number of '_' (which can all be used to move in the direction
        # that maximizes the distance from the origin)
        return abs(count_L - count_R) + count_blank