class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        wildcard_count = moves.count('_')
        
        if left_count >= right_count:
            return wildcard_count + left_count - right_count
        else:
            return wildcard_count + right_count - left_count