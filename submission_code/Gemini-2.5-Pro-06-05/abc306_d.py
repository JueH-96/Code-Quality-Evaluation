import sys

def solve():
    """
    Solves the Wired Full-Course Meal problem using dynamic programming with O(1) space.
    """
    # Read the number of courses.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle empty input.
        return

    # Use a large negative integer to represent impossible states.
    # The total tastiness can be large, so this must be sufficiently small.
    # -10^18 is a safe value for "negative infinity".
    NEG_INF = -10**18

    # dp_healthy: max tastiness achievable ending in a healthy state.
    # dp_upset: max tastiness achievable ending in an upset state.
    
    # Initially, we have a healthy stomach with 0 tastiness.
    # The upset state is initially impossible.
    dp_healthy = 0
    dp_upset = NEG_INF

    # Process each course one by one.
    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())

        if X == 0:  # Antidotal course
            # To be healthy: max of (skip, eat from healthy, eat from upset).
            new_dp_healthy = max(dp_healthy, dp_healthy + Y, dp_upset + Y)
            # To be upset: must have been upset and skipped.
            new_dp_upset = dp_upset
        else:  # Poisonous course
            # To be healthy: must have been healthy and skipped.
            new_dp_healthy = dp_healthy
            # To be upset: max of (skip, eat from healthy).
            new_dp_upset = max(dp_upset, dp_healthy + Y)

        # Update the DP states for the next iteration.
        dp_healthy = new_dp_healthy
        dp_upset = new_dp_upset

    # The final answer is the maximum tastiness achievable while surviving.
    print(max(dp_healthy, dp_upset))

solve()