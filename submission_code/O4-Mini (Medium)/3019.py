class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count fixed moves and wildcards
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_U = moves.count('_')
        
        # Maximize distance by assigning all '_' to right or all to left
        # Position if all '_' go right: +count_R + count_U - count_L
        max_right = count_R + count_U - count_L
        # Position if all '_' go left: count_R - (count_L + count_U)
        max_left = count_R - (count_L + count_U)
        
        # Return the larger absolute distance
        return max(abs(max_right), abs(max_left))