class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        num_L = moves.count('L')
        num_R = moves.count('R')
        num_underscore = moves.count('_')

        max_final_position = num_R + num_underscore - num_L
        min_final_position = num_R - num_L - num_underscore

        return max(abs(max_final_position), abs(min_final_position))