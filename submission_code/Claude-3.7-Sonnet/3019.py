class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count each type of move
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # Calculate the net direction from fixed moves
        net_movement = count_R - count_L
        
        # To maximize distance, add all underscores to the dominant direction
        # or any direction if net_movement is 0
        return abs(net_movement) + count_underscore