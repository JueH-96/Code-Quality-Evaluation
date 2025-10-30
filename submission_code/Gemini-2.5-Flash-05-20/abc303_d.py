# YOUR CODE HERE
import sys

def solve():
    # Read X, Y, Z from the first line
    X, Y, Z = map(int, sys.stdin.readline().split())
    # Read the target string S from the second line
    S = sys.stdin.readline().strip()

    # dp_off: minimum time to type characters processed so far, ending with Caps Lock OFF
    # dp_on: minimum time to type characters processed so far, ending with Caps Lock ON

    # Initial state: no characters typed.
    # Caps Lock is initially OFF. Cost is 0.
    dp_off = 0
    # To be in Caps Lock ON state (before typing any character), we must press Caps Lock once.
    # Cost is Z.
    dp_on = Z

    # Iterate through each character in the string S
    for char_s in S:
        # Store the current dp values before updating for the next character
        prev_dp_off = dp_off
        prev_dp_on = dp_on

        # Determine the cost of typing the current character based on the desired character
        # and the Caps Lock state.
        if char_s == 'a':
            # If we want to type 'a':
            #   - If Caps Lock is OFF, press 'a' key (cost X)
            #   - If Caps Lock is ON, press 'a' key + Shift (cost Y)
            cost_type_char_if_caps_off = X
            cost_type_char_if_caps_on = Y
        else: # char_s == 'A'
            # If we want to type 'A':
            #   - If Caps Lock is OFF, press 'a' key + Shift (cost Y)
            #   - If Caps Lock is ON, press 'a' key (cost X)
            cost_type_char_if_caps_off = Y
            cost_type_char_if_caps_on = X

        # Calculate the new dp_off (minimum cost to type char_s and end with Caps Lock OFF)
        # Option 1: Current state was OFF, we stayed OFF.
        #   Cost: prev_dp_off + cost_to_type_char_if_caps_off
        # Option 2: Current state was ON, we toggled to OFF (cost Z).
        #   Cost: prev_dp_on + Z + cost_to_type_char_if_caps_off
        dp_off = min(prev_dp_off + cost_type_char_if_caps_off,
                     prev_dp_on + Z + cost_type_char_if_caps_off)

        # Calculate the new dp_on (minimum cost to type char_s and end with Caps Lock ON)
        # Option 1: Current state was ON, we stayed ON.
        #   Cost: prev_dp_on + cost_to_type_char_if_caps_on
        # Option 2: Current state was OFF, we toggled to ON (cost Z).
        #   Cost: prev_dp_off + Z + cost_to_type_char_if_caps_on
        dp_on = min(prev_dp_on + cost_type_char_if_caps_on,
                    prev_dp_off + Z + cost_type_char_if_caps_on)

    # After processing all characters, the minimum time is the minimum of ending
    # with Caps Lock OFF or ON.
    print(min(dp_off, dp_on))

# Call the solve function to run the program
solve()