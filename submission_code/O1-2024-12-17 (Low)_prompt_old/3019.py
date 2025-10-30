class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # Option 1: Treat all underscores as "L"
        final_pos_all_left = count_R - count_L - count_underscore
        
        # Option 2: Treat all underscores as "R"
        final_pos_all_right = (count_R + count_underscore) - count_L
        
        return max(abs(final_pos_all_left), abs(final_pos_all_right))