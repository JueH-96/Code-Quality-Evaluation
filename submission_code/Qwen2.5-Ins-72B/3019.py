class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the number of 'L', 'R', and '_' in the moves string
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # Calculate the net distance by considering the difference between 'L' and 'R'
        net_distance = abs(count_L - count_R)
        
        # The furthest distance is the net distance plus the number of underscores
        furthest_distance = net_distance + count_underscore
        
        return furthest_distance