class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        """
        Calculates the distance from the origin of the furthest point you can get to.

        The strategy is to find the net displacement from the fixed 'L' and 'R' moves,
        and then use all the flexible '_' moves to maximize the magnitude of this displacement.

        Let count_R be the number of 'R' moves, count_L be the number of 'L' moves,
        and count_underscore be the number of '_' moves.

        The net position from the fixed moves is count_R - count_L.
        The '_' moves can all be used to either increase or decrease the position.
        To maximize the distance from the origin, we add the count of '_' moves to the
        absolute value of the net position from fixed moves.

        For example, if we have more 'R's than 'L's, the net position is positive.
        To get further away, we should use all '_' as 'R's.
        If we have more 'L's, the net position is negative. To get further away,
        we should use all '_' as 'L's.

        This logic leads to the formula: abs(count_R - count_L) + count_underscore.
        """
        
        # Count the occurrences of each character.
        count_R = moves.count('R')
        count_L = moves.count('L')
        count_underscore = moves.count('_')
        
        # The furthest distance is the absolute difference between 'R' and 'L' moves,
        # plus all the wildcard moves used to extend that distance.
        return abs(count_R - count_L) + count_underscore