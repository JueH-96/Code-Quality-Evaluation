import sys
from collections import defaultdict

MOD = 998244353

# Define state transitions: (prev_state, char) -> new_state
# States are represented as integers 0-3 based on (diff_AB % 2, diff_BC % 2)
# 0: (0,0) - (A-B is even, B-C is even) -> all counts same parity
# 1: (0,1) - (A-B is even, B-C is odd)
# 2: (1,0) - (A-B is odd, B-C is even)
# 3: (1,1) - (A-B is odd, B-C is odd)

# Transition for character 'A': (x, y) -> ((x+1)%2, y)
TRANSITIONS_A = {0:2, 1:3, 2:0, 3:1}
# Transition for character 'B': (x, y) -> ((x+1)%2, (y+1)%2)
TRANSITIONS_B = {0:3, 1:2, 2:1, 3:0}
# Transition for character 'C': (x, y) -> (x, (y+1)%2)
TRANSITIONS_C = {0:1, 1:0, 2:3, 3:2}

# Map characters to their transition dictionaries for easier lookup
CHAR_TRANSITIONS = {
    'A': TRANSITIONS_A,
    'B': TRANSITIONS_B,
    'C': TRANSITIONS_C
}

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # dp[i][s_curr][c_tuple] stores the number of ways to fill S[0...i-1]
    # s_curr: The state P(i) (i.e., (cnt_A-cnt_B)%2, (cnt_B-cnt_C)%2 for prefix S[0...i-1])
    # c_tuple: A tuple (c0,c1,c2,c3) where cx is the count of how many times
    #          P(j) (for j in [0, i]) has been in state x.
    # The sum c0+c1+c2+c3 for dp[i] will always be i+1.
    
    # Initialize dp table. `dp[i]` is a dict where keys are `s_curr`
    # and values are dicts where keys are `c_tuple` and values are counts.
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(N + 1)]
    
    # Base case: before processing any character (i=0), the only prefix state is P(0)=0 (empty string).
    # So, state 0 has appeared once, others zero times.
    dp[0][0][(1,0,0,0)] = 1 

    # Iterate through each position in the input string S
    for i in range(N): # i represents the current character index S[i] to be processed
        # Iterate over all possible previous P-states (P(i))
        for prev_s_val in range(4):
            # Iterate over all possible count tuples and their ways for the previous state
            for prev_c_tuple, ways in dp[i][prev_s_val].items():
                if ways == 0: # Skip if no ways lead to this state
                    continue
                
                # Determine which characters can be placed at S[i]
                # If S[i] is '?', any of 'A', 'B', 'C' can be placed.
                # Otherwise, only S[i] itself can be placed.
                chars_to_try = ['A', 'B', 'C']
                if S[i] != '?':
                    chars_to_try = [S[i]]

                for char_val in chars_to_try:
                    # Calculate the new P-state P(i+1) based on P(i) (prev_s_val) and char_val
                    curr_s_val = CHAR_TRANSITIONS[char_val][prev_s_val]
                    
                    # Update the counts of P-states for the new prefix S[0...i]
                    curr_c_list = list(prev_c_tuple)
                    curr_c_list[curr_s_val] += 1
                    curr_c_tuple = tuple(curr_c_list) # Convert back to tuple for dict key
                    
                    # Add ways to the next DP state (dp[i+1])
                    dp[i+1][curr_s_val][curr_c_tuple] = (dp[i+1][curr_s_val][curr_c_tuple] + ways) % MOD

    total_valid_strings = 0

    # After processing all N characters (string S' is fully formed),
    # iterate through all final states in dp[N] to count valid strings.
    for s_final in range(4): # Iterate over possible final P-states (P(N))
        # Iterate over all final count tuples (c0,c1,c2,c3) and their ways
        for c_final_tuple, ways in dp[N][s_final].items():
            if ways == 0: # Skip if no ways lead to this state
                continue
            
            # Calculate the total number of good substrings for this particular P-state profile.
            # A substring S'[l:r] (0-indexed, half-open) is good if P(r) == P(l).
            # This means we count pairs (j, k) such that j < k and P(j) == P(k).
            # If a state X appears C_X times in the sequence P(0), P(1), ..., P(N),
            # then it contributes C_X * (C_X - 1) / 2 good substrings.
            num_good_sub = 0
            for c_val in c_final_tuple:
                num_good_sub += c_val * (c_val - 1) // 2
            
            # If the number of good substrings is at least K, add the ways to the total count.
            if num_good_sub >= K:
                total_valid_strings = (total_valid_strings + ways) % MOD

    print(total_valid_strings)

solve()