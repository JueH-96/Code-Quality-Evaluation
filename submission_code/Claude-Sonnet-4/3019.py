class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_count = moves.count('L')
        right_count = moves.count('R')
        blank_count = moves.count('_')
        
        # Net position from forced moves
        net_position = right_count - left_count
        
        # To maximize distance from origin, add all blanks in the direction
        # that increases the absolute value of position
        max_distance = abs(net_position) + blank_count
        
        return max_distance