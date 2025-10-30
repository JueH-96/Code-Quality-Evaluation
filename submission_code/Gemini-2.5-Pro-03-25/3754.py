import math

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        """
        Calculates the maximum Manhattan distance from the origin achievable at any time step,
        given a path string s and allowed k modifications.

        The Manhattan Distance between two cells (x_i, y_i) and (x_j, y_j) is |x_i - x_j| + |y_i - y_j|.
        The distance from the origin (0,0) to (x, y) is |x - 0| + |y - 0| = |x| + |y|.

        We want to find the maximum value of |x'_t| + |y'_t| over all possible modified strings s' 
        (obtained from s by changing at most k characters) and all time steps t (0 <= t <= n), 
        where (x'_t, y'_t) is the position after t steps following the path defined by s'.

        Args:
          s: The string representing the sequence of moves ('N', 'S', 'E', 'W').
          k: The maximum number of characters allowed to change.

        Returns:
          The maximum Manhattan distance achievable from the origin at any point during the path.
        """
        n = len(s)
        
        # Step 1: Compute the path coordinates and Manhattan distances for the original string s.
        # This gives us the baseline path without any modifications.
        x, y = 0, 0
        # positions list stores (x, y) coordinates at each time step t=0...n
        # positions[t] = (x_t, y_t), the position after t moves according to the original string s.
        positions = [(0, 0)] 
        for i in range(n):
            move = s[i]
            if move == 'N':
                y += 1
            elif move == 'S':
                y -= 1
            elif move == 'E':
                x += 1
            elif move == 'W': # move == 'W'
                x -= 1
            positions.append((x, y))
        
        # manhattan_distances list stores |x_t| + |y_t| for each time step t=0...n
        # manhattan_distances[t] is the Manhattan distance from the origin at time t for the original path.
        manhattan_distances = [abs(px) + abs(py) for px, py in positions]
        
        # Step 2: Calculate the maximum achievable distance at each time step t, considering k modifications.
        # The maximum achievable distance will be the maximum over all time steps t.
        max_achieved_dist = 0
        
        for t in range(n + 1):
            # M_t0 is the Manhattan distance at time t for the original path.
            M_t0 = manhattan_distances[t]
            
            # Calculate the maximum potential Manhattan distance achievable at time t using at most k changes.
            # Key insight: Each change of a character s[i] to s'[i] (where i < t) can change the final 
            # position (x_t, y_t). The maximum change in Manhattan distance from a single character change
            # is 2 (e.g., 'N' to 'S' changes y by -2).
            # Therefore, k changes can increase the Manhattan distance by at most 2*k.
            # This provides an upper bound on the modified distance: M'_t <= M_t0 + 2*k.
            potential_max_dist_t = M_t0 + 2 * k
            
            # There are two fundamental constraints on the achievable Manhattan distance D' at time t:
            # 1. D' <= t: The Manhattan distance from the origin cannot exceed the number of steps taken.
            #    This is because each step increases the Manhattan distance by at most 1.
            # 2. D' % 2 == t % 2: The Manhattan distance must have the same parity as the number of steps t.
            #    This is because each step changes exactly one coordinate by +/- 1, so x+y changes parity.
            #    The Manhattan distance |x| + |y| has the same parity as x+y.
            #    Since the initial position (0,0) has x+y=0 (even), after t steps, x_t + y_t has the same parity as t.
            #    Therefore, |x_t| + |y_t| must have the same parity as t.

            # Apply constraint 1: The achievable distance is limited by the number of steps t.
            current_max_dist_t = min(potential_max_dist_t, t)
            
            # Apply constraint 2: Adjust for parity.
            # If the calculated maximum possible distance `current_max_dist_t` does not have the same parity as t,
            # then the actual maximum achievable distance respecting the parity constraint must be smaller.
            # The largest value <= `current_max_dist_t` with the correct parity is `current_max_dist_t - 1`.
            if current_max_dist_t % 2 != t % 2:
                 # Ensure the distance does not become negative after decrementing.
                 # This handles the edge case where current_max_dist_t is 0 and t is odd.
                 # The distance must be non-negative.
                 current_max_dist_t = max(0, current_max_dist_t - 1)
            
            # Update the overall maximum distance found across all time steps considered so far.
            max_achieved_dist = max(max_achieved_dist, current_max_dist_t)
            
        # Step 3: Return the overall maximum achievable distance found.
        return max_achieved_dist