class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the number of children in a cycle (round trip)
        cycle_length = n - 1
        
        # Find the remainder of the total seconds when divided by the cycle length
        # This is the effective position of the ball after k seconds, due to its periodic movement.
        effective_pos = k % (2 * cycle_length)
        
        # If the ball's effective position is less than the cycle length,
        # it means it didn't complete a round trip and is still moving right.
        if effective_pos < cycle_length:
            return effective_pos
        
        # If it's equal to the cycle length, it means it's at position n-1.
        # Otherwise, we calculate the position after completing part of a leftward move,
        # which is the symmetric position with respect to n - 1.
        return cycle_length - (effective_pos - cycle_length)