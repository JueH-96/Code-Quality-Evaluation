class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the occurrences of 'L', 'R', and '_' in the moves string
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # Calculate the net displacement ignoring the underscores
        net = count_R - count_L
        
        # The underscores can be used to move in either direction.
        # The furthest distance is the maximum absolute value we can get
        # by adding all underscores to one side or the other.
        return max(abs(net + count_underscore), abs(net - count_underscore))