class Solution:
  def numberOfChild(self, n: int, k: int) -> int:
    """
    Calculates the final position of the ball after k seconds using a mathematical approach.

    The children are numbered from 0 to n - 1.
    The ball starts at child 0, moves right towards n - 1.
    Upon reaching either end (0 or n - 1), the direction reverses.
    Each pass takes 1 second.

    The movement pattern is periodic:
    0 -> 1 -> ... -> n-1 (takes n-1 seconds)
    n-1 -> n-2 -> ... -> 0 (takes n-1 seconds)
    Total time for one full cycle (0 -> n-1 -> 0) is (n-1) + (n-1) = 2 * (n - 1) seconds.

    We can determine the position of the ball after k seconds by considering
    the time k modulo the cycle length. This gives the effective time within a single cycle.

    Args:
      n: The number of children (constraints: 2 <= n <= 50).
      k: The number of seconds elapsed (constraints: 1 <= k <= 50).

    Returns:
      The index of the child holding the ball after k seconds.
    """
    
    # Calculate the period of the ball's movement cycle.
    # Since n >= 2, n - 1 >= 1, and the period is at least 2.
    # If n = 2, period = 2 * (2 - 1) = 2. (0 -> 1 -> 0 -> ...)
    # If n = 3, period = 2 * (3 - 1) = 4. (0 -> 1 -> 2 -> 1 -> 0 -> ...)
    period = 2 * (n - 1)
    
    # Find the effective time within the current cycle using the modulo operator.
    # This tells us where in the cycle the process is after k seconds.
    # time_in_cycle will be in the range [0, period - 1].
    time_in_cycle = k % period
    
    # Determine the position based on the effective time within the cycle.
    
    # The cycle consists of two phases:
    # Phase 1: Moving right from 0 to n-1. This takes n-1 seconds.
    #          This corresponds to time_in_cycle values from 0 to n-1.
    # Phase 2: Moving left from n-1 to 0. This takes n-1 seconds.
    #          This corresponds to time_in_cycle values from n to period-1.
    
    # Check if the time falls into the first phase (moving right).
    if time_in_cycle <= n - 1:
        # In this phase, the position increases linearly from 0.
        # After `t` seconds in this phase, the position is `t`.
        # So, the position is equal to time_in_cycle.
        return time_in_cycle
    else:
        # If the time is beyond n-1, it falls into the second phase (moving left).
        # The ball reached the peak position (n-1) at time n-1 within the cycle.
        # Now it moves left, decreasing its position.
        # The time elapsed since starting the leftward movement is time_in_cycle - (n - 1).
        # The position is the peak position (n-1) minus the steps taken to the left.
        # Position = (n - 1) - (time_in_cycle - (n - 1))
        # Simplifying this formula:
        # Position = n - 1 - time_in_cycle + n - 1
        # Position = 2 * (n - 1) - time_in_cycle
        # Which is also equal to: Position = period - time_in_cycle
        return period - time_in_cycle