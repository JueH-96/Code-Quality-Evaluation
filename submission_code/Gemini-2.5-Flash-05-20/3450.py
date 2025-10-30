class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the length of one full cycle of ball passing.
        # A cycle involves the ball moving from child 0 to child (n-1)
        # and then back from child (n-1) to child 0.
        #
        # It takes (n - 1) seconds to move from 0 to n-1.
        # It takes another (n - 1) seconds to move from n-1 back to 0.
        # So, the total time for one full cycle is 2 * (n - 1) seconds.
        # For n=2 (children 0, 1): cycle is 0 -> 1 -> 0, taking 2 * (2-1) = 2 seconds.
        # For n=3 (children 0, 1, 2): cycle is 0 -> 1 -> 2 -> 1 -> 0, taking 2 * (3-1) = 4 seconds.
        
        cycle_length = 2 * (n - 1)
        
        # Determine the effective number of seconds within the current cycle.
        # This tells us where in the cycle the ball is, irrespective of how many
        # full cycles have been completed.
        effective_k = k % cycle_length
        
        # If effective_k is less than (n - 1):
        # This means the ball is currently in the "forward" (left-to-right) movement phase.
        # The child's number is simply equal to the number of seconds passed in this phase.
        # Example: n=5, k=2. effective_k = 2. Since 2 < (5-1)=4, child is 2.
        if effective_k < n - 1:
            return effective_k
        
        # If effective_k is greater than or equal to (n - 1):
        # This means the ball has reached child (n-1) and is now in the "backward"
        # (right-to-left) movement phase, or has just reached child (n-1).
        #
        # The ball took (n - 1) seconds to reach child (n - 1).
        # The number of steps taken from (n - 1) towards 0 is:
        # `steps_from_end = effective_k - (n - 1)`
        #
        # The current child's position is child (n - 1) minus these steps:
        # `current_child = (n - 1) - steps_from_end`
        # `current_child = (n - 1) - (effective_k - (n - 1))`
        # `current_child = n - 1 - effective_k + n - 1`
        # `current_child = 2 * (n - 1) - effective_k`
        # This is equivalent to `cycle_length - effective_k`.
        #
        # Example: n=5, k=6. effective_k = 6. Since 6 >= (5-1)=4, we use this logic.
        # steps_from_end = 6 - (5 - 1) = 6 - 4 = 2.
        # current_child = (5 - 1) - 2 = 4 - 2 = 2.
        # Alternatively, using the simplified formula: 2 * (5 - 1) - 6 = 8 - 6 = 2.
        else:
            return cycle_length - effective_k