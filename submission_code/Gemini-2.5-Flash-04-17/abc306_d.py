import sys

# Function to solve the problem
def solve():
    # Read the number of courses
    N = int(sys.stdin.readline())

    # dp0: maximum tastiness ending in a healthy state
    # dp1: maximum tastiness ending in an upset state
    # Initialize dp1 to negative infinity, representing an unreachable state
    dp0 = 0
    dp1 = -float('inf') # Use -float('inf') to represent unreachable states with a value smaller than any possible sum.

    # Iterate through each course
    for _ in range(N):
        # Read course type (X) and tastiness (Y)
        X, Y = map(int, sys.stdin.readline().split())

        # Calculate potential DP states after considering the current course
        next_dp0 = -float('inf')
        next_dp1 = -float('inf')

        # --- Calculate potential next_dp0 (ending healthy) ---

        # Option 1: Arrive at healthy by skipping the current course
        # This is possible only if the previous state was healthy.
        if dp0 > -float('inf'):
            next_dp0 = max(next_dp0, dp0)

        # Option 2: Arrive at healthy by eating the current course
        if X == 0: # Current course is Antidotal (type 0)
            # Eating antidotal from previous healthy state (dp0) leads to healthy.
            if dp0 > -float('inf'):
                next_dp0 = max(next_dp0, dp0 + Y)
            # Eating antidotal from previous upset state (dp1) leads to healthy.
            if dp1 > -float('inf'):
                next_dp0 = max(next_dp0, dp1 + Y)
        # If X == 1 (Poisonous), eating leads to upset or death, never healthy.


        # --- Calculate potential next_dp1 (ending upset) ---

        # Option 1: Arrive at upset by skipping the current course
        # This is possible only if the previous state was upset.
        if dp1 > -float('inf'):
            next_dp1 = max(next_dp1, dp1)

        # Option 2: Arrive at upset by eating the current course
        if X == 1: # Current course is Poisonous (type 1)
            # Eating poisonous from previous healthy state (dp0) leads to upset.
            if dp0 > -float('inf'):
                next_dp1 = max(next_dp1, dp0 + Y)
            # Eating poisonous from previous upset state (dp1) leads to death.
            # We do not update next_dp0 or next_dp1 in this case as it's an invalid path.


        # Update DP states for the next iteration
        dp0 = next_dp0
        dp1 = next_dp1

    # After processing all courses, the maximum tastiness achieved while surviving
    # is the maximum of the tastiness values in the final reachable states (healthy or upset).
    # If both final states are unreachable (-inf), it means survival was impossible except by skipping everything.
    # The initial dp0 = 0 covers the "skip everything" case.
    
    # The maximum tastiness achievable while surviving is the max of the two states.
    # If both dp0 and dp1 are -inf, it means no course could be eaten without dying,
    # but skipping everything is always possible with tastiness 0.
    # If either dp0 or dp1 is a valid number, we take the maximum of the two.
    # If both are -inf, max(-inf, -inf) is -inf.
    
    result = max(dp0, dp1)

    # The problem asks for the maximum possible sum of tastiness, or 0 if he eats nothing.
    # Skipping everything results in 0 tastiness. If the calculated `result` is negative,
    # it means the best path involves negative tastiness. In this case, skipping everything (tastiness 0)
    # is better.
    print(max(0, result))

# Call the solve function to run the program
solve()