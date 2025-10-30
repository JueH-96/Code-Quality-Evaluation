class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left = moves.count('L')
        right = moves.count('R')
        underscore = moves.count('_')
        
        return abs(left - right) + underscore