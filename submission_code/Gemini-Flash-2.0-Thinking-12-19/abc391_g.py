import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # State is represented by an integer index from 0 to 2^N - 1.
    # The index corresponds to the binary representation of the sequence of increments (v_i - v_{i-1}).
    # state_idx = sum_{i=1}^N (v_i - v_{i-1}) * 2^(N-i)
    # v = (v_0, v_1, ..., v_N) where v_0=0, v_i - v_{i-1} in {0,1}
    # v_i = LCS(S[0...i-1], T) for a string T

    def decode_state(state_idx, N):
        v = [0] * (N + 1)
        v[0] = 0
        for i in range(1, N + 1):
            # The i-th bit from the left (most significant) corresponds to v_i - v_{i-1}
            # The bit value is at position N-i (0-indexed from the right)
            bit = (state_idx >> (N - i)) & 1
            v[i] = v[i-1] + bit
        return tuple(v)

    def encode_state(v_vec, N):
        state_idx = 0
        for i in range(1, N + 1):
            bit = v_vec[i] - v_vec[i-1] # This difference must be 0 or 1
            state_idx |= (bit << (N - i))
        return state_idx

    # Precompute transitions
    # transitions[current_state_idx][char_code] = next_state_idx
    transitions = [[0] * 26 for _ in range(1 << N)]

    for current_state_idx in range(1 << N):
        v = list(decode_state(current_state_idx, N)) # decode returns tuple, list is mutable
        
        for char_code in range(26):
            c = chr(ord('a') + char_code)
            
            v_prime = [0] * (N + 1)
            v_prime[0] = 0
            current_v_prime_i_minus_1 = 0
            
            for i in range(1, N + 1):
                # S is 0-indexed, v and v_prime are associated with S[0...i-1]
                if S[i-1] == c:
                    # LCS(S[0...i-1], T + c) when S[i-1] == c
                    # This is LCS(S[0...i-2], T) + 1 = v[i-1] + 1
                    v_prime[i] = v[i-1] + 1
                else:
                    # LCS(S[0...i-1], T + c) when S[i-1] != c
                    # This is max(LCS(S[0...i-2], T+c), LCS(S[0...i-1], T))
                    # = max(v_prime[i-1], v[i])
                    v_prime[i] = max(current_v_prime_i_minus_1, v[i])
                current_v_prime_i_minus_1 = v_prime[i]
            
            transitions[current_state_idx][char_code] = encode_state(v_prime, N)

    # DP
    # dp_curr[state_idx] = count of strings of length j ending in this state
    dp_tables = [[0] * (1 << N), [0] * (1 << N)] # Use two lists
    dp_curr_idx = 0
    dp_next_idx = 1

    # Base case: empty string (length 0), state is (0,0,...,0), index 0
    dp_tables[dp_curr_idx][0] = 1

    # Iterate M times for string length
    for j in range(M):
        # Clear the next table by creating a new one (or zeroing out the old one)
        dp_tables[dp_next_idx] = [0] * (1 << N)
        
        for current_state_idx in range(1 << N):
            count = dp_tables[dp_curr_idx][current_state_idx]
            if count == 0:
                continue
                
            for char_code in range(26):
                next_state_idx = transitions[current_state_idx][char_code]
                dp_tables[dp_next_idx][next_state_idx] = (dp_tables[dp_next_idx][next_state_idx] + count) % MOD
        
        # Swap current and next indices for the next iteration
        dp_curr_idx, dp_next_idx = dp_next_idx, dp_curr_idx

    # After M steps, the result is in dp_tables[dp_curr_idx]
    final_dp = dp_tables[dp_curr_idx]

    # Final step: Sum up counts based on the final LCS length (v_N)
    ans = [0] * (N + 1)

    for state_idx in range(1 << N):
        count = final_dp[state_idx]
        if count == 0:
            continue
            
        v = decode_state(state_idx, N)
        lcs_length = v[N] # The last element v_N is LCS(S[0...N-1], T[0...M-1])
        
        if lcs_length <= N: # Should always be true by property of LCS
            ans[lcs_length] = (ans[lcs_length] + count) % MOD

    # Print results
    print(*ans)

solve()