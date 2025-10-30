import sys

def solve():
    MOD = 998244353
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    num_states = 1 << N
    dp = [[0] * num_states for _ in range(2)] # Use 2 rows for space optimization

    # Initial state: empty string T (length 0).
    # dp[i][0] = 0 for all i. The state is (0, 0, ..., 0).
    # The state tuple (v0, v1, ..., vN) where v_i = dp[i][0].
    # The index for state (0,0,...,0) is 0 because all differences v_{i+1}-v_i are 0.
    dp[0][0] = 1

    current_row = 0
    # Iterate through string length j from 0 to M-1. Calculate DP for length j+1.
    for j in range(M):
        next_row = (current_row + 1) % 2
        # Clear the next row
        for k in range(num_states):
            dp[next_row][k] = 0

        # Iterate through all possible previous states at length j (column j)
        for prev_state_idx in range(num_states):
            if dp[current_row][prev_state_idx] == 0:
                continue # No strings of length j result in this state, so no transitions from here

            # Compute the previous state tuple prev_v from prev_state_idx
            # prev_v = (v_0, v_1, ..., v_N) where v_i = dp[i][j]
            prev_v = [0] * (N + 1)
            prev_v[0] = 0
            # The index encodes differences: idx = sum (v_{i+1}-v_i) * 2^i for i=0..N-1
            # Thus, v_{i+1} - v_i = (prev_state_idx >> i) & 1
            # We can compute v_i iteratively: v_i = v_{i-1} + (v_i - v_{i-1})
            v_curr_val = 0
            prev_v[0] = v_curr_val
            for i in range(N): # i is the index for the difference bit (0..N-1)
                diff = (prev_state_idx >> i) & 1 # diff corresponds to v_{i+1} - v_i
                v_next_val = v_curr_val + diff # This is the value of v_{i+1}
                prev_v[i+1] = v_next_val
                v_curr_val = v_next_val
            # prev_v is now [v_0, v_1, ..., v_N] corresponding to dp[0..N][j]

            # Iterate over possible next characters 'a' through 'z' for T[j]
            for char_code in range(26):
                c = chr(ord('a') + char_code)

                # Compute the next state tuple next_v for length j+1 (column j+1)
                # next_v = (v'_0, v'_1, ..., v'_N) where v'_i = dp[i][j+1]
                next_v = [0] * (N + 1)
                next_v[0] = 0 # v'_0 = dp[0][j+1] is always 0

                # Use current_v'_i_minus_1 to build the next_v tuple iteratively
                # current_v_prime_i_minus_1 holds v'_{i-1} = dp[i-1][j+1]
                current_v_prime_i_minus_1 = 0

                # Compute v'_i = dp[i][j+1] for i = 1, ..., N
                for i in range(1, N + 1):
                    # This uses the standard LCS DP transition rules:
                    # dp[i][j+1] = max(dp[i-1][j+1], dp[i][j+1-1]) if S[i-1] != T[j]
                    # dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j+1-1] + 1) if S[i-1] == T[j]
                    # Here T[j] is the character 'c' we are considering.
                    # dp[i-1][j+1] is current_v_prime_i_minus_1.
                    # dp[i][j] is prev_v[i].
                    # dp[i-1][j] is prev_v[i-1].
                    if ord(S[i-1]) - ord('a') == char_code: # If S[i-1] matches char c
                        # Corresponds to S[i-1] == T[j]
                        # v'_i = max(v'_{i-1}, prev_v_{i-1} + 1)
                        next_v[i] = max(current_v_prime_i_minus_1, prev_v[i-1] + 1)
                    else: # If S[i-1] does not match char c
                        # Corresponds to S[i-1] != T[j]
                        # v'_i = max(v'_{i-1}, prev_v_i)
                        next_v[i] = max(current_v_prime_i_minus_1, prev_v[i])

                    current_v_prime_i_minus_1 = next_v[i] # Update v'_{i-1} for the next iteration (i -> i+1)

                # Compute the next state index from next_v tuple
                next_state_idx = 0
                # The index is sum (v'_{i+1} - v'_i) * 2^i for i=0..N-1
                for i in range(N): # i is the index of the difference bit (0..N-1)
                    # The diff bit i corresponds to v'_{i+1} - v'_i
                    next_state_idx += (next_v[i+1] - next_v[i]) * (1 << i)

                # Update DP table: add the count of strings reaching prev_state_idx
                dp[next_row][next_state_idx] = (dp[next_row][next_state_idx] + dp[current_row][prev_state_idx]) % MOD

        current_row = next_row # Move to the next column (length j+1)

    # After iterating through all lengths up to M, the results are in dp[current_row]
    # Collect answers ans[k] for k = 0, 1, ..., N
    # The number of strings T of length M whose LCS with S has length exactly k.
    # The LCS length is dp[N][M], which is the last element v_N in the state tuple (v_0, ..., v_N)
    # associated with a state index at length M.
    # The state tuple (v_0, ..., v_N) is constructed such that v_0=0 and v_{i+1} = v_i + ((idx >> i) & 1).
    # Summing the differences: v_N = v_0 + sum_{i=0}^{N-1} (v_{i+1} - v_i) = 0 + sum_{i=0}^{N-1} ((idx >> i) & 1).
    # This sum is the total number of set bits (population count) in the binary representation of the index 'idx'.
    ans = [0] * (N + 1)
    for state_idx in range(num_states):
        # The value v_N is the population count of the index idx.
        k = bin(state_idx).count('1') # Calculate population count (number of set bits)
        ans[k] = (ans[k] + dp[current_row][state_idx]) % MOD

    # Print the answers ans_0, ans_1, ..., ans_N
    print(*ans)

solve()