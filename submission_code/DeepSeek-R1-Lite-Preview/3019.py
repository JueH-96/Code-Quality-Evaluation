class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # Scenario 1: All '_' are 'L'
        total_L1 = count_L + count_underscore
        total_R1 = count_R
        distance1 = abs(total_R1 - total_L1)
        
        # Scenario 2: All '_' are 'R'
        total_R2 = count_R + count_underscore
        total_L2 = count_L
        distance2 = abs(total_R2 - total_L2)
        
        # Return the maximum distance
        return max(distance1, distance2)