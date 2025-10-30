import sys

def solve():
    """
    Reads input, solves the problem using dynamic programming, and prints the result.
    """
    # Read the number of monsters, N, and their strengths, A.
    # The problem constraints guarantee N >= 1, so no special handling for N=0 is needed.
    try:
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # This handles potential empty input lines, which might occur in
        # some testing environments. For typical contest platforms, input is well-formed.
        return

    # dp_even: max score having defeated an EVEN number of monsters.
    # Initialize to 0, representing the starting state (0 monsters defeated, 0 xp).
    dp_even = 0
    
    # dp_odd: max score having defeated an ODD number of monsters.
    # Initialize to negative infinity, as it's an impossible state at the beginning.
    dp_odd = -float('inf')

    # Iterate through each monster.
    for strength in A:
        # Use temporary variables to store the next state values, ensuring that
        # calculations for the current step are based on the states from the previous step.
        
        # Calculate the new max score for an EVEN number of defeats.
        # This is the max of:
        # 1. Not defeating the monster (score remains dp_even).
        # 2. Defeating it after having an odd number of defeats (score becomes dp_odd + 2*strength).
        new_dp_even = max(dp_even, dp_odd + 2 * strength)

        # Calculate the new max score for an ODD number of defeats.
        # This is the max of:
        # 1. Not defeating the monster (score remains dp_odd).
        # 2. Defeating it after having an even number of defeats (score becomes dp_even + strength).
        new_dp_odd = max(dp_odd, dp_even + strength)
        
        # Update the states for the next iteration.
        dp_even = new_dp_even
        dp_odd = new_dp_odd
        
    # The final answer is the maximum score achievable, which could be from
    # either an even or an odd total number of defeats.
    print(max(dp_even, dp_odd))

solve()