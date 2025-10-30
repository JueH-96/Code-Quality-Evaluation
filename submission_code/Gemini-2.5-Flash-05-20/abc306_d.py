import sys

def solve():
    N = int(sys.stdin.readline())
    
    # dp_healthy: maximum tastiness ending with a healthy stomach
    # dp_upset: maximum tastiness ending with an upset stomach
    # Initialize dp_healthy to 0. This accounts for the case where
    # Takahashi skips all courses, ending with a healthy stomach and 0 tastiness.
    # Initialize dp_upset to -float('inf') because Takahashi cannot start with an upset stomach.
    dp_healthy = 0
    dp_upset = -float('inf')

    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())

        # Create temporary variables for the next state's DP values.
        # Initialize them with the current DP values. This effectively
        # represents the option of "skipping" the current course,
        # where the state and tastiness remain unchanged from the previous step.
        next_dp_healthy = dp_healthy
        next_dp_upset = dp_upset

        # Now, consider the option of "eating" the current course (X, Y)

        # Case 1: Eating the course while starting with a Healthy stomach
        if dp_healthy != -float('inf'): # Ensure the healthy state was reachable
            if X == 0: # Antidotal course (0): Healthy -> Healthy
                # If eating this antidotal course leads to a higher tastiness for a healthy stomach, update.
                next_dp_healthy = max(next_dp_healthy, dp_healthy + Y)
            else: # Poisonous course (1): Healthy -> Upset
                # If eating this poisonous course leads to a higher tastiness for an upset stomach, update.
                next_dp_upset = max(next_dp_upset, dp_healthy + Y)

        # Case 2: Eating the course while starting with an Upset stomach
        if dp_upset != -float('inf'): # Ensure the upset state was reachable
            if X == 0: # Antidotal course (0): Upset -> Healthy
                # If eating this antidotal course leads to a higher tastiness for a healthy stomach, update.
                next_dp_healthy = max(next_dp_healthy, dp_upset + Y)
            else: # Poisonous course (1): Upset -> Dies.
                # This option is forbidden as it leads to death. We do not update any DP values for this path.
                pass 
        
        # Update the main DP states for the next iteration using the calculated next_dp values.
        dp_healthy = next_dp_healthy
        dp_upset = next_dp_upset

    # After processing all N courses, Takahashi must not be dead.
    # The maximum tastiness is the maximum of ending in a healthy or an upset state.
    # Since dp_healthy starts at 0 and can always be maintained at least 0 by skipping
    # courses, the result will always be non-negative.
    print(max(dp_healthy, dp_upset))

solve()