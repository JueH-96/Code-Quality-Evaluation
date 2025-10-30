# YOUR CODE HERE
import sys

# Set higher recursion depth if needed, although the solution uses iteration.
# import sys
# sys.setrecursionlimit(200010) 

# Faster input reading function
# input = sys.stdin.readline 

def solve():
    # Read input values for N, M, K
    N, M, K = map(int, sys.stdin.readline().split())
    
    # Store shortcut edges. Each element is a list [X_i, Y_i].
    shortcuts = []
    for i in range(M):
        shortcuts.append(list(map(int, sys.stdin.readline().split()))) 

    # Define the modulus for calculations
    MOD = 998244353

    # Handle the base case where M=0 (no shortcuts).
    # In this case, the graph is just a single cycle. Starting from vertex 1,
    # there is exactly one unique path of length K following the cycle edges.
    if M == 0:
        print(1)
        return

    # Helper function to calculate distance along the cycle.
    # d(u, v) = number of cycle edges to traverse from vertex u to vertex v.
    # Vertices are 1-based indexed.
    def cycle_dist(u, v):
        # The distance is (v - u) mod N.
        # Example: N=6. d(5, 2) = (2-5)%6 = -3%6 = 3. Path: 5->6->1->2 (3 steps).
        # Python's % operator handles negative results correctly for positive N,
        # producing results in [0, N-1].
        dist = (v - u) % N
        return dist

    # Precompute costs for transitions between abstract states.
    # State i (0 <= i < M): represents the state immediately after taking the i-th shortcut 
    # (from the input list, 0-indexed) which ends at vertex Y_i.
    
    # costs_0i: Cost to transition from the initial state (at vertex 1) to state i by taking the first shortcut.
    # The cost is defined as the number of cycle steps from vertex 1 to X_i, plus 1 for the shortcut edge itself.
    costs_0i = {} # Map: shortcut_idx (0..M-1) -> cost
    for i in range(M):
        Xi, Yi = shortcuts[i] # Get start (X_i) and end (Y_i) vertices of the i-th shortcut
        cost = (cycle_dist(1, Xi) + 1) # Calculate cost: distance along cycle + 1 step for shortcut
        costs_0i[i] = cost # Store the cost associated with using shortcut i as the first shortcut

    # costs_ji: Cost to transition from state j to state i.
    # This represents taking shortcut j, ending at Y_j, then traversing cycle edges to X_i, and finally taking shortcut i.
    # The cost is the number of cycle steps from Y_j to X_i, plus 1 for shortcut i.
    costs_ji = {} # Map: (from_shortcut_idx j, to_shortcut_idx i) -> cost
    for j in range(M):
        Yj = shortcuts[j][1] # End vertex of shortcut j
        for i in range(M):
            Xi = shortcuts[i][0] # Start vertex of shortcut i
            cost = (cycle_dist(Yj, Xi) + 1) # Calculate cost: distance along cycle + 1 step for shortcut i
            costs_ji[(j, i)] = cost # Store the cost for transition j -> i

    # Dynamic Programming state definition:
    # dp_values[k] is a dictionary { i: count }
    # 'count' is the number of sequences of shortcuts ending with the i-th shortcut,
    # such that the total "cost" of the sequence is exactly k.
    # The "cost" of a sequence of shortcuts is the total path length covered by the cycle segments 
    # preceding each shortcut and the shortcuts themselves.
    dp_values = {} 

    # Initialize DP table with base cases: sequences using exactly one shortcut.
    # For each shortcut i, if its initial cost c_0i is within K, it forms a valid sequence start.
    for i in range(M):
        cost = costs_0i[i]
        if cost <= K: # Only consider sequences whose total cost does not exceed K
            # Ensure the dictionary for this cost 'k' exists
            if cost not in dp_values:
                dp_values[cost] = {}
            # Increment the count for sequences ending with shortcut i at this cost
            # Use .get(i, 0) to handle cases where i is not yet a key, defaulting to 0.
            dp_values[cost][i] = (dp_values[cost].get(i, 0) + 1) % MOD

    # Iterate through costs k from 1 up to K.
    # This loop builds the DP table iteratively based on previously computed values.
    for k in range(1, K + 1):
        # Retrieve the DP states for the current cost k. If none exist, skip.
        # Use .get(k, {}) which returns an empty dict if k is not found, avoiding KeyError.
        current_dp_at_k = dp_values.get(k, {})
        if not current_dp_at_k: continue # If dictionary is empty, no sequences reach cost k, so skip

        # For each shortcut index j that ends a sequence with cost k:
        for j in current_dp_at_k: 
            count_at_j = current_dp_at_k[j] # Number of sequences ending in j with cost k
            if count_at_j == 0: continue # If count is 0, no sequences to extend

            # Consider extending these sequences by transitioning from state j to state i using shortcut i.
            for i in range(M):
                key_ji = (j, i) # Key for the transition cost from j to i
                # Retrieve the cost for transition j -> i. Use .get to handle missing entries safely.
                cost_ji = costs_ji.get(key_ji) 
                
                # Check if a valid transition cost exists
                if cost_ji is not None: 
                    new_k = k + cost_ji # Calculate the total cost of the extended sequence
                    
                    # If the new total cost is within the limit K:
                    if new_k <= K:
                        # Ensure the dictionary for the new cost 'new_k' exists
                        if new_k not in dp_values:
                            dp_values[new_k] = {}
                        
                        # Add the count of sequences ending at j (count_at_j) to the count for sequences ending at i with cost new_k
                        dp_values[new_k][i] = (dp_values[new_k].get(i, 0) + count_at_j) % MOD

    # Summing up the results:
    # The total number of paths of length exactly K is:
    # 1 (for the unique path using only cycle edges)
    # + Sum over all k from 1 to K, and all shortcut indices i, of dp_values[k][i].
    # Each dp_values[k][i] represents sequences of shortcuts whose structure uses k steps.
    # The remaining K-k steps must be cycle edges after the last shortcut Y_i. This is always possible and unique.
    
    total_sum = 1 # Initialize with 1 for the path using only cycle edges
    
    # Iterate through all costs k for which DP entries were computed
    for k in dp_values: 
         # Get the dictionary of counts for this cost k
         current_dp_at_k = dp_values[k]
         # Iterate through all shortcut indices i that end sequences with cost k
         for i in current_dp_at_k: 
             # Add the count of such sequences to the total sum
             total_sum = (total_sum + current_dp_at_k[i]) % MOD

    # Print the final total count modulo MOD
    print(total_sum)

# Execute the solve function
solve()