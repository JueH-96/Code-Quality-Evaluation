# YOUR CODE HERE
import sys

def solve():
    # Read input values for X, Y, Z
    # X: cost to press 'a' key
    # Y: cost to press Shift + 'a' keys
    # Z: cost to press Caps Lock key
    X, Y, Z = map(int, sys.stdin.readline().split())
    
    # Read the target string S
    S = sys.stdin.readline().strip()
    N = len(S)
    
    # Use a large integer constant for infinity. 
    # The maximum possible cost can be roughly N * max(X, Y, Z), 
    # which can be up to 3e5 * 1e9 = 3e14. 
    # 10**18 is sufficiently large.
    INF = 10**18 

    # Dynamic Programming states:
    # dp0: minimum cost to type the prefix S[:i] ending with Caps Lock OFF
    # dp1: minimum cost to type the prefix S[:i] ending with Caps Lock ON
    
    # Initialize DP states for the empty prefix (length 0)
    # Base case: To type an empty string, the cost is 0. 
    # The initial state is Caps Lock OFF.
    dp0 = 0 
    
    # To reach the state (empty string, Caps Lock ON) from the initial state,
    # we must press the Caps Lock key once. The cost is Z.
    dp1 = Z 

    # Iterate through the string character by character from index 0 to N-1
    for i in range(N):
        # The character we need to type at this step
        char_to_type = S[i]
        
        # Temporary variables to store the minimum costs for the next state (after typing S[i])
        # Initialize with infinity to ensure minimum is correctly found across different paths
        new_dp0 = INF 
        new_dp1 = INF

        # Case 1: The character to type is 'a'
        if char_to_type == 'a':
            # Calculate minimum cost to type 'a' and end with Caps Lock OFF (new_dp0)
            
            # Path 1.1: Arriving from previous state OFF (cost dp0). 
            # To type 'a' with Caps Lock OFF, press 'a' key (cost X). State remains OFF.
            cost_from_0_to_0 = dp0 + X
            
            # Path 1.2: Arriving from previous state ON (cost dp1).
            # Need to type 'a' and end with Caps Lock OFF. This requires toggling Caps Lock (cost Z).
            # We need to consider two sequences of actions:
            # Option A: Press Caps Lock (Z) -> state becomes OFF. Then press 'a' key (X). Total cost Z+X.
            # Option B: Press Shift+'a' key (Y) (types 'a' because Caps Lock is ON) -> state remains ON. Then press Caps Lock (Z) -> state becomes OFF. Total cost Y+Z.
            # The minimum cost for this path is the minimum of these two options.
            cost_from_1_to_0 = dp1 + min(Z + X, Y + Z)
            
            # Update minimum cost to end OFF for current character
            new_dp0 = min(cost_from_0_to_0, cost_from_1_to_0)

            # Calculate minimum cost to type 'a' and end with Caps Lock ON (new_dp1)

            # Path 2.1: Arriving from previous state OFF (cost dp0).
            # Need to type 'a' and end with Caps Lock ON. This requires toggling Caps Lock (cost Z).
            # Option A: Press Caps Lock (Z) -> state becomes ON. Then press Shift+'a' key (Y) (types 'a'). Total cost Z+Y.
            # Option B: Press 'a' key (X) (types 'a') -> state remains OFF. Then press Caps Lock (Z) -> state becomes ON. Total cost X+Z.
            # Take the minimum.
            cost_from_0_to_1 = dp0 + min(Z + Y, X + Z)
            
            # Path 2.2: Arriving from previous state ON (cost dp1).
            # To type 'a' with Caps Lock ON, press Shift+'a' key (Y). State remains ON.
            cost_from_1_to_1 = dp1 + Y
            
            # Update minimum cost to end ON for current character
            new_dp1 = min(cost_from_0_to_1, cost_from_1_to_1)

        # Case 2: The character to type is 'A'
        else: # char_to_type == 'A'
            # Calculate minimum cost to type 'A' and end with Caps Lock OFF (new_dp0)

            # Path 1.1: Arriving from previous state OFF (cost dp0).
            # To type 'A' with Caps Lock OFF, press Shift+'a' key (Y). State remains OFF.
            cost_from_0_to_0 = dp0 + Y

            # Path 1.2: Arriving from previous state ON (cost dp1).
            # Need to type 'A' and end with Caps Lock OFF. Requires toggling Caps Lock (Z).
            # Option A: Press Caps Lock (Z) -> state OFF. Then press Shift+'a' key (Y). Total cost Z+Y.
            # Option B: Press 'a' key (X) (types 'A') -> state ON. Then press Caps Lock (Z) -> state OFF. Total cost X+Z.
            # Take the minimum.
            cost_from_1_to_0 = dp1 + min(Z + Y, X + Z)
            
            # Update minimum cost to end OFF
            new_dp0 = min(cost_from_0_to_0, cost_from_1_to_0)

            # Calculate minimum cost to type 'A' and end with Caps Lock ON (new_dp1)

            # Path 2.1: Arriving from previous state OFF (cost dp0).
            # Need to type 'A' and end with Caps Lock ON. Requires toggling Caps Lock (Z).
            # Option A: Press Caps Lock (Z) -> state ON. Then press 'a' key (X). Total cost Z+X.
            # Option B: Press Shift+'a' key (Y) -> state OFF. Then press Caps Lock (Z) -> state ON. Total cost Y+Z.
            # Take the minimum.
            cost_from_0_to_1 = dp0 + min(Z + X, Y + Z)

            # Path 2.2: Arriving from previous state ON (cost dp1).
            # To type 'A' with Caps Lock ON, press 'a' key (X). State remains ON.
            cost_from_1_to_1 = dp1 + X

            # Update minimum cost to end ON
            new_dp1 = min(cost_from_0_to_1, cost_from_1_to_1)
        
        # Update DP states for the next character iteration. 
        # The new states become the current states for the next step.
        dp0 = new_dp0
        dp1 = new_dp1

    # After processing all characters in S, the final answer is the minimum cost 
    # to produce the entire string S. This could end with Caps Lock either ON or OFF.
    # So we take the minimum of the two final states.
    print(min(dp0, dp1))

# Execute the solve function to run the program
solve()