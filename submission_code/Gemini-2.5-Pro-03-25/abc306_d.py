# YOUR CODE HERE
import sys

# Use float('inf') for representing negative infinity effectively.
# This is useful for initialization and handling unreachable states in max operations.
neg_inf = -float('inf')

def solve():
    """
    Solves the problem using dynamic programming.
    Tracks the maximum tastiness achievable ending in either a healthy state (dp0)
    or an upset state (dp1) after considering each course.
    """
    N = int(sys.stdin.readline())
    
    # Initialize DP states
    # dp0 represents the maximum tastiness ending in a healthy state after considering previous courses.
    # dp1 represents the maximum tastiness ending in an upset state after considering previous courses.
    dp0 = 0  # Start healthy with 0 tastiness initially.
    dp1 = neg_inf # It's impossible to start with an upset stomach. Initialize to negative infinity.

    # Iterate through each course from 1 to N
    for _ in range(N):
        # Read the type (X) and tastiness (Y) of the current course
        X, Y = map(int, sys.stdin.readline().split())
        
        # Store current DP values. The calculation for the current course (step i)
        # must use the DP values from the previous step (i-1).
        current_dp0 = dp0
        current_dp1 = dp1

        # Initialize DP values for the current step (after considering course i)
        # to negative infinity. This ensures that if a state is unreachable, 
        # its value remains neg_inf and doesn't wrongly affect the maximum calculation.
        new_dp0 = neg_inf
        new_dp1 = neg_inf

        # Process based on the type of course (Antidotal X=0 or Poisonous X=1)
        if X == 0: # Antidotal course
            # Calculate the maximum tastiness ending in a healthy state after this course.
            
            # Consider transitions from the previous healthy state (current_dp0)
            if current_dp0 != neg_inf: # Check if the healthy state was reachable
                 # Option 1: Skip the current course. State remains healthy. Tastiness doesn't change.
                 new_dp0 = max(new_dp0, current_dp0)
                 # Option 2: Eat the current course. State remains healthy. Tastiness increases by Y.
                 new_dp0 = max(new_dp0, current_dp0 + Y)
            
            # Consider transitions from the previous upset state (current_dp1)
            if current_dp1 != neg_inf: # Check if the upset state was reachable
                 # Option 3: Eat the current course. State becomes healthy. Tastiness increases by Y.
                 new_dp0 = max(new_dp0, current_dp1 + Y)
                 # Option 4: Skip the course - State remains upset. Does not contribute to new_dp0.

            # Calculate the maximum tastiness ending in an upset state after this course.
            # After an antidotal course, the only way to end up upset is if you started upset and skipped the course.
            if current_dp1 != neg_inf: # Check if the upset state was reachable
                 # Option: From upset state, skip the course. State remains upset. Tastiness doesn't change.
                 new_dp1 = max(new_dp1, current_dp1) 

        else: # X == 1, Poisonous course
            # Calculate the maximum tastiness ending in a healthy state after this course.
            # After a poisonous course, the only way to end up healthy is if you started healthy and skipped the course.
            if current_dp0 != neg_inf: # Check if the healthy state was reachable
                 # Option: From healthy state, skip the course. State remains healthy. Tastiness doesn't change.
                 new_dp0 = max(new_dp0, current_dp0)

            # Calculate the maximum tastiness ending in an upset state after this course.
            
            # Consider transitions from the previous healthy state (current_dp0)
            if current_dp0 != neg_inf: # Check if the healthy state was reachable
                 # Option 1: Eat the current course. State becomes upset. Tastiness increases by Y.
                 new_dp1 = max(new_dp1, current_dp0 + Y)
                 # Option 2: Skip the course - State remains healthy. Does not contribute to new_dp1.
            
            # Consider transitions from the previous upset state (current_dp1)
            if current_dp1 != neg_inf: # Check if the upset state was reachable
                 # Option 3: Skip the course. State remains upset. Tastiness doesn't change.
                 new_dp1 = max(new_dp1, current_dp1)
                 # Option 4: Eat the course - Takahashi dies. This path is invalid and not considered.

        # Update DP states for the next iteration with the newly calculated maximum tastiness values.
        dp0 = new_dp0
        dp1 = new_dp1
        
    # After processing all N courses, the final answer is the maximum possible tastiness
    # achievable while surviving until the end. This could be ending in either a healthy state (dp0)
    # or an upset state (dp1). We take the maximum of the two possibilities.
    # The DP formulation naturally handles the case "or 0 if he eats nothing" because
    # the initial state dp0=0 represents 0 tastiness, and skipping courses maintains this path.
    # If all eating paths result in negative total tastiness, max(dp0, dp1) will choose the 0 path if it exists.
    print(max(dp0, dp1))

# Call the solve function to execute the program logic.
solve()