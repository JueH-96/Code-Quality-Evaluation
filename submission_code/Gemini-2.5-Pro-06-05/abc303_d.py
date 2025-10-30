# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Keyboard problem using dynamic programming with space optimization.
    """
    # Read input values
    # X: cost to press 'a' key
    # Y: cost to press Shift + 'a'
    # Z: cost to press Caps Lock
    try:
        X, Y, Z = map(int, sys.stdin.readline().split())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Gracefully handle empty input lines at the end of file
        return

    # dp_off: min cost to have typed the prefix so far, ending with Caps OFF.
    # dp_on: min cost to have typed the prefix so far, ending with Caps ON.

    # Base case: Before typing any characters.
    # The keyboard starts with Caps Lock OFF.
    # To be in the initial state (Caps OFF), the cost is 0.
    dp_off = 0
    # To switch to Caps ON from the initial state, the cost is Z.
    dp_on = Z

    # Iterate through each character of the string S to build up the solution.
    for char in S:
        if char == 'a':
            # Cost to type 'a' with Caps OFF is X (press 'a').
            # Cost to type 'a' with Caps ON is Y (press Shift + 'a').
            cost_char_off = X
            cost_char_on = Y
        else:  # char == 'A'
            # Cost to type 'A' with Caps OFF is Y (press Shift + 'a').
            # Cost to type 'A' with Caps ON is X (press 'a').
            cost_char_off = Y
            cost_char_on = X
        
        # Calculate the new DP states for the current character.
        # We use temporary variables to ensure we use the states from the *previous* step
        # for both calculations in this iteration.
        
        # Calculate the new minimum cost to end with Caps Lock OFF.
        # This is the minimum of two possibilities:
        # 1. Coming from a state where Caps was already OFF.
        # 2. Coming from a state where Caps was ON, then switching it OFF (cost Z).
        new_dp_off = min(dp_off + cost_char_off, dp_on + Z + cost_char_off)
        
        # Calculate the new minimum cost to end with Caps Lock ON.
        # This is the minimum of two possibilities:
        # 1. Coming from a state where Caps was already ON.
        # 2. Coming from a state where Caps was OFF, then switching it ON (cost Z).
        new_dp_on = min(dp_on + cost_char_on, dp_off + Z + cost_char_on)
        
        # Update the states for the next iteration.
        dp_off = new_dp_off
        dp_on = new_dp_on

    # After processing the entire string, the final answer is the minimum
    # of the two possible final states (ending with Caps Lock OFF or ON).
    print(min(dp_off, dp_on))

solve()