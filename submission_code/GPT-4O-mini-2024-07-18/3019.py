class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_moves = moves.count('L')
        right_moves = moves.count('R')
        underscore_moves = moves.count('_')
        
        # The furthest distance can be calculated by considering all possible moves
        # We can move left for all 'L' and '_' and right for all 'R' and '_'
        # Thus, the furthest distance is the sum of all left and right potential moves
        return left_moves + right_moves + underscore_moves