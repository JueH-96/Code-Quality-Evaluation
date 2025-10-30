import math

class Solution:
  def furthestDistanceFromOrigin(self, moves: str) -> int:
    """
    Calculates the maximum distance from the origin achievable after a sequence of moves,
    interpreting the requirement based on consistency with provided examples.

    Args:
        moves: A string representing the sequence of moves, containing 'L', 'R', and '_'.
               'L' means move left (-1), 'R' means move right (+1), 
               '_' means choose to move either left (-1) or right (+1).

    Returns:
        The maximum absolute distance from the origin after n moves. Based on analyzing
        the examples, the intended calculation appears to be the maximum of the 
        absolute values of the two extreme reachable points (all '_' as 'L' vs all '_' as 'R').
        This simplifies to abs(count_R - count_L) + count_underscore.
        
        Alternatively, if strictly following the example outputs (especially Example 1 where output=3),
        the calculation might be abs(min_position), where min_position is achieved by treating all '_' as 'L'.
        min_position = count_R - (count_L + count_underscore). Let's verify this interpretation again.
        
        Interpretation 1: Maximize absolute distance.
        Max positive position (all '_' as R): pos_max = (count_R + count_underscore) - count_L
        Min negative position (all '_' as L): pos_min = count_R - (count_L + count_underscore)
        Result = max(abs(pos_max), abs(pos_min))
        This simplifies to: abs(count_R - count_L) + count_underscore
        Ex1: "L_RL__R" -> L=1, R=2, _=4. abs(2-1)+4 = 1+4 = 5. (Example output: 3)
        Ex2: "_R__LL_" -> L=2, R=1, _=4. abs(1-2)+4 = 1+4 = 5. (Example output: 5)
        Ex3: "_______" -> L=0, R=0, _=7. abs(0-0)+7 = 0+7 = 7. (Example output: 7)
        This interpretation matches Ex2 and Ex3, but not Ex1's output.

        Interpretation 2: Result is abs(min_position)
        Result = abs(count_R - (count_L + count_underscore))
        Ex1: "L_RL__R" -> abs(2 - (1 + 4)) = abs(2 - 5) = abs(-3) = 3. (Example output: 3)
        Ex2: "_R__LL_" -> abs(1 - (2 + 4)) = abs(1 - 6) = abs(-5) = 5. (Example output: 5)
        Ex3: "_______" -> abs(0 - (0 + 7)) = abs(0 - 7) = abs(-7) = 7. (Example output: 7)
        This interpretation matches all example outputs.

        Given the ambiguity in the problem statement vs. Example 1, but consistency
        of Interpretation 1 with the natural reading ("furthest") and Examples 2 & 3,
        it's likely Example 1's output/explanation might be flawed. The standard interpretation
        of "furthest distance" implies maximizing the absolute value. Let's implement Interpretation 1.
    """
    
    # Count the occurrences of L, R, and _
    count_L = moves.count('L')
    count_R = moves.count('R')
    count_underscore = moves.count('_')
    
    # Calculate the base displacement from fixed moves ('L' and 'R')
    base_displacement = count_R - count_L
    
    # The maximum distance from the origin is achieved by using all '_' moves
    # to amplify the magnitude of the base displacement.
    # This means adding the count of '_' to the absolute base displacement.
    # If base_displacement is positive, treat all '_' as 'R'. Final pos = base_displacement + count_underscore. Distance = base_displacement + count_underscore.
    # If base_displacement is negative, treat all '_' as 'L'. Final pos = base_displacement - count_underscore. Distance = abs(base_displacement - count_underscore) = -(base_displacement - count_underscore) = -base_displacement + count_underscore = abs(base_displacement) + count_underscore.
    # If base_displacement is zero, treat all '_' as R or L yields distance count_underscore = abs(0) + count_underscore.
    # In all cases, max_distance = abs(base_displacement) + count_underscore.
    max_distance = abs(base_displacement) + count_underscore
    
    return max_distance