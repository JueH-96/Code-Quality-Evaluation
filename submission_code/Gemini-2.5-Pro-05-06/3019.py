class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        """
        Calculates the furthest distance from the origin after a series of moves.

        Args:
          moves: A string of length n consisting only of 'L', 'R', and '_'.

        Returns:
          The distance from the origin of the furthest point you can get to.
        """
        
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')
        
        # The net displacement if we only consider 'L' and 'R' moves.
        # 'R' moves contribute +1 to position, 'L' moves contribute -1.
        base_displacement = count_R - count_L
        
        # Each '_' can be chosen to be 'L' (-1) or 'R' (+1).
        # To maximize the distance from the origin, we want to maximize |final_position|.
        # The two extreme positions are achieved when all '_' characters are chosen
        # to move in the same direction.
        #
        # Position 1 (all '_' are 'R'): base_displacement + count_underscore
        # Position 2 (all '_' are 'L'): base_displacement - count_underscore
        #
        # The furthest distance is max(abs(Position 1), abs(Position 2)).
        # This simplifies to abs(base_displacement) + count_underscore,
        # because count_underscore is non-negative.
        #
        # This represents the magnitude of the fixed displacement,
        # augmented by all underscores contributing to move further in that direction
        # (or an arbitrary direction if fixed displacement is zero).
        
        furthest_distance = abs(base_displacement) + count_underscore
        
        return furthest_distance