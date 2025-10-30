import sys
from collections import defaultdict

MOD = 998244353

# Function to compute the next DP state tuple
# The state is represented by a tuple (dp_j[1], ..., dp_j[N])
# Given the current state tuple v_tuple and the next character char,
# this function computes the next state tuple (dp_{j+1}[1], ..., dp_{j+1}[N])
# based on the LCS DP recurrence.
def compute_next_state(v_tuple, char, S):
    N = len(S)
    # v_tuple represents (dp_j[1], ..., dp_j[N]).
    # We need (dp_j[0], ..., dp_j[N]) for the recurrence, where dp_j[0] = 0.
    # We create a list v where v[i] = dp_j[i] for i = 0...N.
    v = [0] + list(v_tuple) 
    
    # v_prime will store dp_{j+1}[0], ..., dp_{j+1}[N]
    v_prime = [0] * (N + 1)
    
    # current_v_prime will store dp_{j+1}[i-1] as we compute dp_{j+1}[i] in the loop
    current_v_prime = 0 
    
    # Compute dp_{j+1}[i] for i = 1...N using dp_{j+1}[i-1] and dp_j[i] or dp_j[i-1]
    for i in range(1, N + 1):
        # S is 0-indexed: S[0] ... S[N-1] corresponds to S_1 ... S_N in math DP
        # v is 0-indexed: v[0] ... v[N] represents dp_j[0] ... dp_j[N]
        # v_prime is 0-indexed: v_prime[0] ... v_prime[N] stores dp_{j+1}[0] ... dp_{j+1}[N]
        
        if S[i-1] == char:
            # Recurrence if S[i] == char: dp_{j+1}[i] = max(dp_{j+1}[i-1], dp_j[i-1] + 1)
            # In our indexing: max(v_prime[i-1], v[i-1] + 1)
            X_i = v[i-1] + 1 
        else:
            # Recurrence if S[i] != char: dp_{j+1}[i] = max(dp_{j+1}[i-1], dp_j[i])
            # In our indexing: max(v_prime[i-1], v[i])
            X_i = v[i] 
            
        # Calculate dp_{j+1}[i] = max(dp_{j+1}[i-1], X_i)
        # current_v_prime holds dp_{j+1}[i-1] from the previous iteration step (i-1)
        v_prime[i] = max(current_v_prime, X_i) 
        
        # Update current_v_prime for the next iteration step (i+1)
        current_v_prime = v_prime[i]
        
    # The next state tuple is (dp_{j+1}[1], ..., dp_{j+1}[N])
    return tuple(v_prime[1:])

def solve():
    # Read input N, M, and S
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Find unique characters in S
    unique_chars = sorted(list(set(S)))
    num_unique = len(unique_chars)
    
    # Find a character not in S, if any exist.
    # Any character not in S will behave the same way in transitions.
    other_char = None
    if num_unique < 26:
        all_chars = set('abcdefghijklmnopqrstuvwxyz')
        # Pick any one character not in S
        other_char = list(all_chars - set(unique_chars))[0] 

    # DP table: maps state tuple (v_1, ..., v_N) to count of strings of length j that result in this state.
    # We use a single dictionary `dp` and update it in each step M times.
    # `dp` stores counts for strings of the current length j.
    # `new_dp` stores counts for strings of length j+1.
    dp = defaultdict(int)
    
    # Initial state: empty string (length 0).
    # The DP state (dp_0[1], ..., dp_0[N]) is (0, ..., 0) because LCS with an empty string is always 0.
    initial_state = tuple([0] * N)
    dp[initial_state] = 1

    # Perform M DP steps
    for _ in range(M):
        new_dp = defaultdict(int)
        
        # Iterate through current states and their counts
        for state, count in dp.items():
            
            # Transitions by appending each unique character present in S
            for char in unique_chars:
                next_state = compute_next_state(state, char, S)
                # Add the count to the new state's count in new_dp, modulo MOD
                new_dp[next_state] = (new_dp[next_state] + count) % MOD
            
            # Transitions by appending a character not in S
            if other_char is not None:
                # All 26 - num_unique characters not in S lead to the same next state.
                next_state_other = compute_next_state(state, other_char, S)
                num_other_chars = 26 - num_unique
                # Add the count * number of such characters to the new state's count, modulo MOD
                new_dp[next_state_other] = (new_dp[next_state_other] + count * num_other_chars) % MOD
                
        # The new_dp becomes the current dp for the next iteration
        dp = new_dp 

    # After M steps, dp contains counts for strings of length M.
    # The final answers are the counts for each possible LCS length k = 0...N.
    # The LCS length for a state (v_1, ..., v_N) is v_N (the last element).
    ans = [0] * (N + 1) # ans[k] will store the count of strings with LCS length k
    
    # Sum up counts from all final states, grouped by their LCS length
    for state, count in dp.items():
        # The last element of the state tuple is v_N = dp_M[N], which is the LCS length
        lcs_length = state[-1] 
        ans[lcs_length] = (ans[lcs_length] + count) % MOD

    # Print the answers for k=0, 1, ..., N
    print(*ans)

# Execute the solve function
solve()