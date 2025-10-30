# YOUR CODE HERE
import sys
from collections import defaultdict

# Set higher recursion depth if needed, although this DP is iterative.
# Using a high value like 2000 for N=50 is safe, although likely unnecessary.
# sys.setrecursionlimit(2000) 

MOD = 998244353

def solve():
    # Read input: N = string length, K = minimum required good substrings
    N, K = map(int, sys.stdin.readline().split())
    # Read the input string S
    S = sys.stdin.readline().strip()

    # The dynamic programming state will be stored in a dictionary `dp`.
    # Keys are tuples representing the state: (k, v_idx, C_tuple)
    # Values are the number of ways to reach that state.
    # `k`: accumulated count of good substrings found so far, capped at K. 
    #      A value of K signifies "at least K" good substrings.
    # `v_idx`: an integer index from 0 to 3 representing the parity vector V[i] = ( (P_A[i]-P_B[i]) % 2, (P_B[i]-P_C[i]) % 2 ).
    #          Mapping: 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1).
    # `C_tuple`: a tuple (C00, C01, C10, C11). Cij stores the count of prefix indices `p` (from 0 to current `i`) 
    #            such that V[p] has the parity corresponding to index `ij`. For example C01 stores count of p where V[p]=(0,1).
    
    dp = defaultdict(int)

    # Base case: Represents the state before processing any characters (conceptually at index i=-1).
    # This corresponds to the empty prefix string "".
    # The number of good substrings accumulated is 0, so k=0.
    # For the empty prefix, we define V[0] = (0,0). This is because prefix sums P_A, P_B, P_C are all 0. 
    # The parity difference vector (P_A[0]-P_B[0] % 2, P_B[0]-P_C[0] % 2) is (0, 0).
    # The integer index for V[0]=(0,0) is v_idx=0.
    # The counts tuple C tracks the frequencies of V values for indices {0}. Only index 0 exists, and V[0]=(0,0). 
    # So C = (1, 0, 0, 0), indicating one occurrence of V=(0,0) and zero occurrences of others.
    # The initial state has 1 way to be reached (the empty prefix configuration).
    dp[(0, 0, (1, 0, 0, 0))] = 1

    # Precompute mappings between V tuple (parity vector) and its integer index for efficiency and clarity.
    v_map = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3} # Tuple to index
    idx_to_v = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)} # Index to tuple

    # Iterate through the input string S character by character, from index i=0 to N-1
    for i in range(N):
        # Prepare a dictionary to store the states reachable after processing character S[i]
        new_dp = defaultdict(int)
        
        # Determine the character options at the current position i
        current_char_spec = S[i] # This could be 'A', 'B', 'C', or '?'
        
        possible_chars = []
        if current_char_spec == '?':
            # If it's '?', we can replace it with 'A', 'B', or 'C'
            possible_chars = ['A', 'B', 'C']
        else:
            # If it's a specific character, that's the only option
            possible_chars = [current_char_spec]

        # Process each state reachable after processing prefix S[0..i-1]
        for state, count in dp.items():
            # If count is 0, this state has no ways to reach it, so it contributes nothing further.
            # defaultdict handles this, but explicitly checking can be a minor optimization.
            # if count == 0: continue 

            # Unpack the current state tuple
            k, v_idx, C_tuple = state
            # Get the V value (parity vector) for the prefix ending at i-1, which is V[i]
            current_V = idx_to_v[v_idx] 

            # Explore transitions for each possible character `char_code` that can be placed at index i
            for char_code in possible_chars:
                
                # Calculate the V value for the prefix ending at i, which is V[i+1].
                # This depends on the previous V value (V[i] = current_V) and the character `char_code`.
                next_V = None
                if char_code == 'A':
                    # Adding 'A' increases P_A by 1. This increases D_AB = P_A - P_B by 1 mod 2. D_BC = P_B - P_C is unchanged mod 2.
                    # Parity V changes from (v0, v1) to ((v0+1)%2, v1)
                    next_V = ( (current_V[0] + 1) % 2, current_V[1] )
                elif char_code == 'B':
                    # Adding 'B' increases P_B by 1. This decreases D_AB by 1 mod 2 (or increases by 1 mod 2). Increases D_BC by 1 mod 2.
                    # Parity V changes from (v0, v1) to ((v0-1)%2, (v1+1)%2) = ((v0+1)%2, (v1+1)%2)
                    next_V = ( (current_V[0] + 1) % 2, (current_V[1] + 1) % 2 )
                else: # char_code == 'C'
                    # Adding 'C' increases P_C by 1. D_AB is unchanged mod 2. Decreases D_BC by 1 mod 2 (or increases by 1 mod 2).
                    # Parity V changes from (v0, v1) to (v0, (v1-1)%2) = (v0, (v1+1)%2)
                    next_V = ( current_V[0], (current_V[1] + 1) % 2 )
                
                # Get the integer index corresponding to the calculated `next_V` tuple
                next_v_idx = v_map[next_V]

                # Calculate the number of new good substrings ending at the current index i.
                # A substring S[j..i] (1-based index j, i+1) is good if V[i+1] == V[j-1]. (0-based index j, i)
                # The number of such starting positions `j` is equal to the count of indices `p` in {0, ..., i} 
                # such that V[p] == V[i+1]. This count is readily available in the `C_tuple` at index `next_v_idx`.
                delta_k = C_tuple[next_v_idx]
                
                # Update the total count of good substrings found so far.
                new_k = k + delta_k
                
                # Cap the accumulated count at K. If `new_k` reaches or exceeds K, we set it to K.
                # The state component `k=K` represents having found "at least K" good substrings.
                capped_new_k = min(new_k, K)

                # Update the C tuple (counts of V values) for the state after processing character S[i].
                # This new tuple C' reflects the counts for prefix indices {0, ..., i+1}.
                # To get C', we take the current C tuple and increment the count corresponding to V[i+1] = `next_V`.
                new_C_list = list(C_tuple) # Convert tuple to list to allow modification
                new_C_list[next_v_idx] += 1
                new_C_tuple = tuple(new_C_list) # Convert back to tuple to use it as a dictionary key

                # Define the resulting state after this transition.
                new_state = (capped_new_k, next_v_idx, new_C_tuple)
                
                # Add the number of ways (`count`) to reach the current state `state` 
                # to the number of ways to reach the `new_state`.
                # Perform this addition modulo MOD. Using defaultdict simplifies this update.
                new_dp[new_state] = (new_dp[new_state] + count) % MOD

        # After considering all current states and transitions, `new_dp` holds the states for the prefix ending at i.
        # Update `dp` for the next iteration (processing character S[i+1]).
        dp = new_dp
        
        # Optional debug print: monitor the size of the DP table to understand performance.
        # print(f"Step {i+1}, dp size: {len(dp)}")

    # After iterating through all N characters of S, the `dp` dictionary contains the final states for the complete strings.
    # The final answer is the sum of counts for all states where the accumulated good substring count `k` is equal to K.
    # Recall that `k=K` represents having found at least K good substrings due to the capping logic.
    total_count = 0
    for state, count in dp.items():
        # Unpack the state tuple; we only need the `k` value for the final sum.
        k, _, _ = state 
        # If the state's accumulated count `k` reached the cap `K`
        if k == K:
            # Add the count of ways to reach this state to the total answer.
            total_count = (total_count + count) % MOD
            
    # Print the final total count modulo MOD.
    print(total_count)

# Execute the solve function to run the program
solve()