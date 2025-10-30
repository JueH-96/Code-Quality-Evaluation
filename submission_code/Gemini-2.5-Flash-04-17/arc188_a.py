import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # Mapping state indices to vectors (parity diffs)
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 2
    # (1,1) -> 3
    def idx_to_vec(p): return (p >> 1, p & 1)
    def vec_to_idx(v): return (v[0] << 1) | v[1]

    # Vectors corresponding to character contributions to parity diffs
    # D('A') -> (1,0)
    # D('B') -> (1,1)
    # D('C') -> (0,1)
    D_vec = {'A': (1,0), 'B': (1,1), 'C': (0,1)}

    # dp[i][p][c0][c1][c2] stores the number of ways to fill S[0...i-1]
    # such that state[i] has type p, and the counts of state types
    # 0, 1, 2 in state[0...i] are c0, c1, c2 respectively.
    # c3 is implicitly (i+1) - c0 - c1 - c2.
    # dp[i][p] is a dictionary mapping (c0, c1, c2) tuple to count.
    dp = [ [{} for _ in range(4)] for _ in range(N + 1) ]

    # Base case: i = 0 (empty prefix S[0...-1])
    # state[0] is (0,0), type 0.
    # Counts for state[0...0] are (1, 0, 0, 0) for types 0, 1, 2, 3.
    # So c0=1, c1=0, c2=0.
    dp[0][0][(1, 0, 0)] = 1

    # Precompute binomial coefficients C(n, 2)
    C_n_2 = [0] * (N + 2)
    for n in range(2, N + 2):
        C_n_2[n] = n * (n - 1) // 2

    # DP transition
    for i in range(N): # Filling S[i]
        for p_curr in range(4): # type of state[i]
            for (c0, c1, c2), count in dp[i][p_curr].items():
                # c3 is implicitly (i + 1) - c0 - c1 - c2 # not needed directly here

                possible_chars = []
                if S[i] == '?':
                    possible_chars = ['A', 'B', 'C']
                else:
                    possible_chars = [S[i]]

                v_curr = idx_to_vec(p_curr)

                for char in possible_chars:
                    d_vec = D_vec[char]
                    # v_next = v_curr XOR d_vec
                    v_next = (v_curr[0] ^ d_vec[0], v_curr[1] ^ d_vec[1])
                    p_next = vec_to_idx(v_next) # type of state[i+1]

                    # New counts for state[0...i+1]
                    # based on counts for state[0...i] (c0, c1, c2, c3)
                    # state[i+1] has type p_next
                    new_counts_list = [c0, c1, c2, (i + 1) - c0 - c1 - c2]
                    new_counts_list[p_next] += 1

                    # Update DP table for i+1
                    new_counts_tuple = (new_counts_list[0], new_counts_list[1], new_counts_list[2])
                    dp[i+1][p_next][new_counts_tuple] = (dp[i+1][p_next].get(new_counts_tuple, 0) + count) % MOD

    # Final calculation
    total_ways = 0
    # After filling S[0...N-1], we are at state i=N.
    # dp[N][p_final][c0][c1][c2] stores counts for state[0...N].
    # Sum of counts c0+c1+c2+c3 = N+1.
    for p_final in range(4): # type of state[N] - not directly used in final sum, but needed for DP structure
        for (c0, c1, c2), count in dp[N][p_final].items():
            c3 = (N + 1) - c0 - c1 - c2
            
            # Calculate total good substrings for this configuration
            # Number of good substrings is sum of C(m_v, 2) over all state types v
            # where m_v is the count of type v in state[0...N].
            # These counts are c0, c1, c2, c3.
            k_total = C_n_2[c0] + C_n_2[c1] + C_n_2[c2] + C_n_2[c3]

            if k_total >= K:
                total_ways = (total_ways + count) % MOD

    print(total_ways)

solve()