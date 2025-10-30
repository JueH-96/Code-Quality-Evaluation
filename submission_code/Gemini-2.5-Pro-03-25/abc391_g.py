# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # The number of states in our DP is 2^N. Each state corresponds to a possible
    # column vector in the standard LCS dynamic programming table. A state represents
    # the LCS lengths between prefixes of S and a prefix of T.
    num_states = 1 << N
    
    # We need mappings between a state ID (an integer k from 0 to 2^N-1) and the
    # corresponding state tuple D = (d_0, d_1, ..., d_N).
    # The tuple D represents (LCS(S[0..-1], T'), LCS(S[0..0], T'), ..., LCS(S[0..N-1], T'))
    # for some prefix T' of T. d_0 is always 0.
    # The property d_i - d_{i-1} is either 0 or 1 holds for these states.
    # We can map state ID k to tuple D such that the i-th bit of k (0-indexed from right)
    # represents the difference d_{i+1} - d_i.
    id_to_state = {}
    state_to_id = {}

    for k in range(num_states):
        D = [0] * (N + 1)
        # Construct the state tuple D from state ID k based on the difference property.
        for i in range(N):
            # The i-th bit (0-indexed) of k determines the difference d_{i+1} - d_i.
            diff = (k >> i) & 1 
            D[i+1] = D[i] + diff
        
        state_tuple = tuple(D)
        id_to_state[k] = state_tuple
        state_to_id[state_tuple] = k

    # Precompute the state transitions.
    # transitions[k][char_ord] stores the ID of the state reached from state k
    # after appending the character corresponding to char_ord (0='a', ..., 25='z').
    transitions = [[0] * 26 for _ in range(num_states)]

    # Iterate through all possible current states k.
    for k in range(num_states):
        current_state_tuple = id_to_state[k] # Get the tuple representation D of state k.
        
        # Iterate through all 26 possible lowercase English characters.
        for char_ord in range(26):
            char = chr(ord('a') + char_ord)
            
            # Compute the next state tuple D' based on the current state tuple D and the character `char`.
            # This computation mimics the standard LCS DP update rule applied column-wise.
            # D = (dp[0][j], ..., dp[N][j]), D' = (dp[0][j+1], ..., dp[N][j+1])
            next_state_list = [0] * (N + 1) # Use a list for efficient updates.
            
            # Compute each element d'_i of D' for i from 1 to N. d'_0 is always 0.
            for i in range(1, N + 1):
                # The recurrence depends on whether S[i] (using 1-based index for S, which is S[i-1] in 0-based index) matches `char`.
                if S[i-1] == char:
                    # If S[i] matches `char`, LCS length increases based on the state without S[i] and without `char`.
                    # The rule is dp[i][j+1] = dp[i-1][j] + 1.
                    # dp[i-1][j] corresponds to current_state_tuple[i-1].
                    next_state_list[i] = current_state_tuple[i-1] + 1
                else:
                    # If S[i] does not match `char`, LCS length is the max of excluding S[i] or excluding `char`.
                    # The rule is dp[i][j+1] = max(dp[i-1][j+1], dp[i][j]).
                    # dp[i-1][j+1] corresponds to the value computed for the previous row 'i-1' in the current column 'j+1', which is next_state_list[i-1].
                    # dp[i][j] corresponds to the value for the current row 'i' in the previous column 'j', which is current_state_tuple[i].
                    next_state_list[i] = max(next_state_list[i-1], current_state_tuple[i])

            next_state_tuple = tuple(next_state_list)
            
            # Find the state ID corresponding to the computed next state tuple D'.
            # Theory guarantees that D' will always correspond to a valid state tuple already mapped.
            next_k = state_to_id[next_state_tuple]
            
            # Store the transition result.
            transitions[k][char_ord] = next_k

    # Dynamic programming to count the number of strings.
    # dp[k] will store the number of strings of the current length that result in state k.
    dp = [0] * num_states
    
    # Base case: The empty string (length 0) corresponds to the initial state (0,0,...,0), which has ID 0. There is 1 such string.
    dp[0] = 1 

    # Iterate M times, once for each position in the target string T of length M.
    for _ in range(M):
        new_dp = [0] * num_states # Temporary array to store counts for the next length.
        
        # Iterate through all possible current states k.
        for k in range(num_states):
            # If the count for the current state k is 0, no strings end in this state, so skip.
            if dp[k] == 0: 
                continue
            
            count = dp[k] # Number of strings of current length ending in state k.
            
            # Consider appending each possible character.
            for char_ord in range(26):
                next_k = transitions[k][char_ord] # Determine the next state ID based on precomputed transitions.
                
                # Add the count of strings ending in state k to the count for the resulting next state `next_k`.
                # Perform addition modulo MOD.
                new_dp[next_k] = (new_dp[next_k] + count) % MOD
        
        # Update the dp table for the next iteration (next length).
        dp = new_dp 

    # After M iterations, dp[k] contains the number of strings of length M resulting in state k.
    # Collect the final answers based on the LCS length associated with each final state.
    # The LCS length for a state D = (d_0, ..., d_N) is d_N.
    ans = [0] * (N + 1) # ans[k] will store the total count for strings with LCS length k.
    
    # Iterate through all possible final states k.
    for k in range(num_states):
        # If the count for state k is 0, it doesn't contribute to the answer.
        if dp[k] == 0: 
            continue
        
        # Get the tuple representation D of the final state k.
        final_state_tuple = id_to_state[k]
        # The LCS length of S and a string T ending in this state is the last element of the tuple, d_N.
        lcs_len = final_state_tuple[N] 
        
        # Add the count of strings ending in state k to the total count for `ans[lcs_len]`.
        # Perform addition modulo MOD.
        ans[lcs_len] = (ans[lcs_len] + dp[k]) % MOD

    # Print the final counts for LCS lengths 0 to N, separated by spaces.
    print(*(ans))

solve()