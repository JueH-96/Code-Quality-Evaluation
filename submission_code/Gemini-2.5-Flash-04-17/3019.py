class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the number of 'L', 'R', and '_' characters in the moves string.
        count_L = moves.count('L')
        count_R = moves.count('R')
        count_underscore = moves.count('_')

        # The moves 'L' and 'R' contribute a fixed amount to the final position.
        # 'R' contributes +1, 'L' contributes -1.
        # The net displacement from these fixed moves is count_R - count_L.
        # Let this be 'base_displacement'.

        # The '_' character can be chosen as either 'L' (-1) or 'R' (+1).
        # To maximize the distance from the origin, we should use all the '_' moves
        # to go further in the direction that is currently favored by the 'R' and 'L' moves.
        #
        # If base_displacement (count_R - count_L) is positive or zero, we should use all '_' as 'R'
        # to increase the position. The final position would be base_displacement + count_underscore.
        # Example: base_displacement = 2, count_underscore = 3. Max positive position = 2 + 3 = 5.
        #
        # If base_displacement (count_R - count_L) is negative, we should use all '_' as 'L'
        # to decrease the position (make it more negative). The final position would be
        # base_displacement - count_underscore.
        # Example: base_displacement = -2, count_underscore = 3. Max negative position = -2 - 3 = -5.
        #
        # The furthest distance from the origin is the maximum of the absolute values
        # of the two potential final positions: (base_displacement + count_underscore)
        # and (base_displacement - count_underscore).
        # The maximum of abs(base_displacement + count_underscore) and abs(base_displacement - count_underscore)
        # is equal to abs(base_displacement) + count_underscore.
        # This is because the count_underscore moves can always be used to add distance
        # away from the origin, regardless of the initial balance from 'R' and 'L'.
        # The absolute difference abs(count_R - count_L) gives the distance from the origin
        # we are forced towards by the mandatory moves. The count of '_' adds the maximum
        # possible extension to this distance, as each '_' can contribute 1 step further
        # in the optimal direction.

        # Calculate the maximum distance using the simplified formula.
        base_displacement = count_R - count_L
        furthest_distance = abs(base_displacement) + count_underscore

        return furthest_distance