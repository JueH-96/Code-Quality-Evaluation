class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        forced_left = moves.count('L')
        forced_right = moves.count('R')
        flexible = moves.count('_')
        
        # net forced movement
        net = forced_right - forced_left
        
        # Add all flexible moves in the direction of net for the max distance
        return abs(net) + flexible