# YOUR CODE HERE
import sys

def solve():
    # Read input values N, R, C
    N, R, C = map(int, sys.stdin.readline().split())
    # Read the wind sequence string S
    S = sys.stdin.readline().strip()

    # Define displacement vectors for each wind direction
    # Using tuples (row_change, column_change)
    moves = {
        'N': (-1, 0), # North: decrease row index
        'W': (0, -1), # West: decrease column index
        'S': (1, 0),  # South: increase row index
        'E': (0, 1)   # East: increase column index
    }

    # Initialize the cumulative displacement W_t. At t=0, W_0 = (0,0).
    current_W = (0, 0) 
    
    # Initialize the set W_G which stores the cumulative displacements W_k
    # for all times k where new smoke was generated (k is in G).
    # At time t=0, smoke is generated, so G={0} and W_G contains W_0 = (0,0).
    # Using a set allows for efficient average O(1) checking of existence (contains) and addition.
    W_G = { (0, 0) }
    
    # Initialize a list to store the result characters ('0' or '1') for each time step.
    ans = []

    # Iterate through time steps t from 1 to N.
    # In each iteration, we calculate state at time t+0.5.
    for t in range(1, N + 1):
        # Get the wind direction character for the current time step t.
        # The string S is 0-indexed, so S[t-1] corresponds to the wind at time t.
        direction = S[t-1]
        # Get the displacement vector (dr, dc) for this direction.
        dr, dc = moves[direction]
        
        # Update the cumulative displacement W_t.
        # W_t = W_{t-1} + V_t, where V_t is the displacement vector at time t.
        # Tuple arithmetic needs element-wise operations.
        current_W = (current_W[0] + dr, current_W[1] + dc) 
        
        # Determine if new smoke is generated at time t.
        # New smoke is generated at time t if the cell (0,0) is empty after the wind blows.
        # This happens if and only if the cumulative displacement W_t has not been reached 
        # at any previous generation time k < t (k in G).
        # In other words, W_t must be different from all W_k for k in G, k < t.
        # The set W_G currently contains {W_k | k in G, k < t}.
        # So we check if current_W (which is W_t) is already in W_G.
        
        is_new_position = current_W not in W_G
        
        if is_new_position:
            # If W_t is a new position among generated packet locations,
            # then time t is a generation time (t is added to G).
            # We update the set W_G by adding W_t.
            W_G.add(current_W)
            # After this conditional block, W_G contains {W_k | k in G, k <= t}.
        # If W_t was already in W_G, it means W_t = W_j for some j in G, j < t.
        # In this case, t is not a generation time, and W_G already effectively contains W_t.
        # Thus, the set W_G always represents {W_k | k in G, k <= current time t} after this block.

        # Check if smoke exists at the target cell (R,C) at time t+0.5.
        # Smoke exists at (R,C) if at least one smoke packet is at (R,C).
        # A smoke packet generated at time k (where k is in G and k <= t)
        # will be at position W_t - W_k at time t+0.5.
        # We need to check if there exists k in G, k <= t such that W_t - W_k = (R,C).
        # This is equivalent to checking if there exists k in G, k <= t such that W_k = W_t - (R,C).
        
        # Calculate the required W_k value based on the current W_t and the target coordinates (R,C).
        # Tuple subtraction needs element-wise operations.
        Target_Wk = (current_W[0] - R, current_W[1] - C)
        
        # Check if this required W_k value exists in the set W_G.
        # W_G contains exactly the set {W_k | k in G, k <= t}.
        if Target_Wk in W_G:
            # If Target_Wk is found in W_G, it means smoke exists at (R,C). Append '1'.
            ans.append('1')
        else:
            # Otherwise, no smoke packet is at (R,C). Append '0'.
            ans.append('0')
            
    # After iterating through all time steps, join the list of characters into a single string.
    # Print the final result string to standard output.
    print("".join(ans))

# Call the solve function to execute the logic.
solve()