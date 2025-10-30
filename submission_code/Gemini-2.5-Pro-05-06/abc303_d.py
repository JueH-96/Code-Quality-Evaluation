import sys

def solve():
    # Read costs X, Y, Z
    X, Y, Z = map(int, sys.stdin.readline().split())
    
    # Read the target string S
    S = sys.stdin.readline().strip()

    # dp_caps_off: Minimum cost to type the prefix processed so far,
    #              ending with Caps Lock OFF.
    # dp_caps_on:  Minimum cost to type the prefix processed so far,
    #              ending with Caps Lock ON.
    
    # Initial state (before typing any characters):
    # To be in Caps Lock OFF state: cost is 0 (this is the initial keyboard state).
    dp_caps_off = 0
    # To be in Caps Lock ON state: must press Caps Lock key once. Cost is Z.
    dp_caps_on = Z

    # Iterate through each character in the string S
    for char_s in S:
        if char_s == 'a':
            # Cost to type 'a' if Caps Lock is OFF for the typing action (press 'a' key)
            cost_for_off_state = X
            # Cost to type 'a' if Caps Lock is ON for the typing action (press Shift + 'a' key)
            cost_for_on_state = Y
        else:  # char_s == 'A'
            # Cost to type 'A' if Caps Lock is OFF for the typing action (press Shift + 'a' key)
            cost_for_off_state = Y
            # Cost to type 'A' if Caps Lock is ON for the typing action (press 'a' key)
            cost_for_on_state = X

        # Calculate next_dp_caps_off:
        # Min cost to type S[0...current_char] and end with Caps Lock OFF.
        # Option 1: Previous state was Caps Lock OFF. Keep Caps Lock OFF.
        #   Cost = current dp_caps_off + cost_for_off_state.
        # Option 2: Previous state was Caps Lock ON. Toggle Caps Lock from ON to OFF (cost Z), then type.
        #   Cost = current dp_caps_on + Z + cost_for_off_state.
        next_dp_caps_off = min(dp_caps_off + cost_for_off_state, 
                               dp_caps_on + Z + cost_for_off_state)

        # Calculate next_dp_caps_on:
        # Min cost to type S[0...current_char] and end with Caps Lock ON.
        # Option 1: Previous state was Caps Lock ON. Keep Caps Lock ON.
        #   Cost = current dp_caps_on + cost_for_on_state.
        # Option 2: Previous state was Caps Lock OFF. Toggle Caps Lock from OFF to ON (cost Z), then type.
        #   Cost = current dp_caps_off + Z + cost_for_on_state.
        next_dp_caps_on = min(dp_caps_on + cost_for_on_state,
                              dp_caps_off + Z + cost_for_on_state)
        
        # Update dp states for the next iteration
        dp_caps_off = next_dp_caps_off
        dp_caps_on = next_dp_caps_on
        
    # After processing all characters, the minimum cost is the minimum of
    # ending in Caps Lock OFF state or Caps Lock ON state.
    print(min(dp_caps_off, dp_caps_on))

# Call the solver function
if __name__ == '__main__':
    solve()