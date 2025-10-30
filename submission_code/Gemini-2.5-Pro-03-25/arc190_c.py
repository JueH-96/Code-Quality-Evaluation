# YOUR CODE HERE
import sys

def main():
    # Read grid dimensions H, W
    H, W = map(int, sys.stdin.readline().split())
    
    # Read grid values into a 2D list A (0-based indexing internal representation)
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    # Read number of queries Q and initial position (sh, sw)
    Q, sh, sw = map(int, sys.stdin.readline().split())
    # Convert to 0-based indexing for internal use
    sh -= 1 
    sw -= 1

    # Define the modulus
    MOD = 998244353

    # Initialize DP tables for prefix and suffix sums of products
    # dp_prefix[r][c] stores the sum of products of values along paths from (0,0) to (r,c)
    dp_prefix = [[0] * W for _ in range(H)]
    # dp_suffix[r][c] stores the sum of products of values along paths from (r,c) to (H-1, W-1)
    dp_suffix = [[0] * W for _ in range(H)]

    # Function to compute the entire DP prefix table
    def compute_full_dp_prefix():
        for r in range(H):
            for c in range(W):
                # Base case: The only path to (0,0) is itself.
                if r == 0 and c == 0:
                    dp_prefix[r][c] = A[r][c]
                else:
                    # Paths to (r,c) must come from (r-1, c) or (r, c-1)
                    term1 = dp_prefix[r-1][c] if r > 0 else 0 # Contribution from above
                    term2 = dp_prefix[r][c-1] if c > 0 else 0 # Contribution from left
                    # Apply the DP recurrence relation
                    dp_prefix[r][c] = (A[r][c] * (term1 + term2)) % MOD

    # Function to compute the entire DP suffix table
    def compute_full_dp_suffix():
        # Iterate grid backwards from (H-1, W-1) up to (0,0)
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, -1, -1):
                 # Base case: The only path from (H-1, W-1) is itself.
                if r == H - 1 and c == W - 1:
                    dp_suffix[r][c] = A[r][c]
                else:
                    # Paths from (r,c) must go to (r+1, c) or (r, c+1)
                    term1 = dp_suffix[r+1][c] if r < H - 1 else 0 # Contribution to paths going down
                    term2 = dp_suffix[r][c+1] if c < W - 1 else 0 # Contribution to paths going right
                    # Apply the DP recurrence relation
                    dp_suffix[r][c] = (A[r][c] * (term1 + term2)) % MOD

    # Function to update a portion of the DP prefix table
    # This recalculates DP values for cells (r, c) where r >= r_start and c >= c_start
    def update_dp_prefix_region(r_start, c_start):
         for r in range(r_start, H):
             for c in range(c_start, W):
                 # The base case update logic is correct because (0,0) is only updated if r_start=0, c_start=0
                 if r == 0 and c == 0:
                     dp_prefix[r][c] = A[r][c]
                 else:
                     term1 = dp_prefix[r-1][c] if r > 0 else 0
                     term2 = dp_prefix[r][c-1] if c > 0 else 0
                     dp_prefix[r][c] = (A[r][c] * (term1 + term2)) % MOD

    # Function to update a portion of the DP suffix table
    # This recalculates DP values for cells (r, c) where r <= r_end and c <= c_end
    def update_dp_suffix_region(r_end, c_end):
         for r in range(r_end, -1, -1):
             for c in range(c_end, -1, -1):
                  # The base case update logic is correct because (H-1, W-1) is only updated if r_end=H-1, c_end=W-1
                 if r == H - 1 and c == W - 1:
                      dp_suffix[r][c] = A[r][c]
                 else:
                      term1 = dp_suffix[r+1][c] if r < H - 1 else 0
                      term2 = dp_suffix[r][c+1] if c < W - 1 else 0
                      dp_suffix[r][c] = (A[r][c] * (term1 + term2)) % MOD


    # Perform initial full computation of DP tables
    compute_full_dp_prefix()
    compute_full_dp_suffix()

    # Map directions to coordinate changes (dr, dc)
    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}

    # List to store results for each query
    results = []

    # Process Q queries
    for i in range(Q):
        # Read query details: direction d_i and new value a_i
        line = sys.stdin.readline().split()
        d_i = line[0]
        a_i = int(line[1])

        # Update Takahashi's position based on direction d_i
        dr, dc = moves[d_i]
        sh += dr
        sw += dc
        
        # Identify the cell (h*, w*) to be updated (0-based indexing)
        h_star, w_star = sh, sw 

        # Calculate the contribution factor T_(h*, w*) using DP tables *before* the update
        # T combines paths arriving at (h*, w*) and paths departing from (h*, w*)
        
        # Calculate DP'_prefix[h*][w*]: sum of products for paths arriving at (h*, w*)
        # This represents the sum of products along paths ending just before cell (h*, w*)
        dp_prime_prefix_val = 0
        if h_star == 0 and w_star == 0:
             # For the start cell (0,0), the preceding path product is conventionally 1
             dp_prime_prefix_val = 1 
        else:
            # Sum contributions from paths ending at the top neighbor (h*-1, w*)
            if h_star > 0:
                dp_prime_prefix_val = (dp_prime_prefix_val + dp_prefix[h_star-1][w_star]) % MOD
            # Sum contributions from paths ending at the left neighbor (h*, w*-1)
            if w_star > 0:
                 dp_prime_prefix_val = (dp_prime_prefix_val + dp_prefix[h_star][w_star-1]) % MOD
        
        # Calculate DP'_suffix[h*][w*]: sum of products for paths departing from (h*, w*)
        # This represents the sum of products along paths starting just after cell (h*, w*)
        dp_prime_suffix_val = 0
        if h_star == H - 1 and w_star == W - 1:
            # For the end cell (H-1, W-1), the succeeding path product is conventionally 1
            dp_prime_suffix_val = 1 
        else:
            # Sum contributions from paths starting at the bottom neighbor (h*+1, w*)
            if h_star < H - 1:
                dp_prime_suffix_val = (dp_prime_suffix_val + dp_suffix[h_star+1][w_star]) % MOD
            # Sum contributions from paths starting at the right neighbor (h*, w*+1)
            if w_star < W - 1:
                dp_prime_suffix_val = (dp_prime_suffix_val + dp_suffix[h_star][w_star+1]) % MOD
        
        # Calculate T value by multiplying prefix and suffix path contributions
        T_val = (dp_prime_prefix_val * dp_prime_suffix_val) % MOD

        # Get the old value A_old at (h*, w*) before updating it
        A_old = A[h_star][w_star]
        # The new value is A_new = a_i
        A_new = a_i
        
        # Calculate the change in total sum (Delta S) due to the update at (h*, w*)
        # Delta S = T * (A_new - A_old)
        # Add MOD before taking modulo to handle potential negative result from (A_new - A_old)
        delta_S = (T_val * (A_new - A_old + MOD)) % MOD
        
        # Calculate the new total sum S
        # The total sum is given by dp_prefix[H-1][W-1] for the current grid state.
        # We update this sum using the calculated delta_S
        current_sum = (dp_prefix[H-1][W-1] + delta_S) % MOD
        
        # Record the result (new total sum) for this query
        results.append(current_sum)

        # Update the grid value A[h*, w*] to the new value A_new
        A[h_star][w_star] = A_new
        
        # Update the DP tables to reflect the change in A[h*, w*]
        # This prepares the DP tables for the next query calculation
        
        # Update the region in dp_prefix table affected by the change: cells (r,c) where r >= h_star and c >= w_star
        update_dp_prefix_region(h_star, w_star)

        # Update the region in dp_suffix table affected by the change: cells (r,c) where r <= h_star and c <= w_star
        update_dp_suffix_region(h_star, w_star)

    # Print all recorded results
    for res in results:
      print(res)

# Execute the main function when the script is run
if __name__ == '__main__':
    main()