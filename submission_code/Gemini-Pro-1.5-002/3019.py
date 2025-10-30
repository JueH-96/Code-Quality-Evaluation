class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        underscore_count = moves.count('_')
        
        max_dist = 0
        
        max_dist = abs(left_count + underscore_count - (right_count + underscore_count))
        
        
        return abs(left_count - right_count) + underscore_count