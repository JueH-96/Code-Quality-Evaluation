class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        """
        Calculates the child who has the ball after k seconds.

        The method uses a mathematical approach based on the periodic movement
        of the ball, providing an O(1) time complexity solution.
        """
        
        # The time it takes for the ball to travel from one end of the line
        # to the other. This is the duration of one pass.
        pass_duration = n - 1

        # Determine the number of full passes completed.
        # A pass is a traversal from one end to the other.
        # e.g., pass 0: 0 -> n-1, pass 1: n-1 -> 0, pass 2: 0 -> n-1, etc.
        num_passes = k // pass_duration
        
        # The remaining time determines the position within the current pass.
        time_in_current_pass = k % pass_duration
        
        # If the number of passes is even, the direction is from left to right.
        if num_passes % 2 == 0:
            # The ball starts at child 0 and moves right.
            # The position is equal to the time elapsed in this pass.
            return time_in_current_pass
        else:
            # If the number of passes is odd, the direction is from right to left.
            # The ball starts at child n-1 and moves left.
            # The position is (n-1) minus the time elapsed in this pass.
            return pass_duration - time_in_current_pass