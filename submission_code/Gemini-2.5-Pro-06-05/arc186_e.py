import sys

def solve():
    """
    Main function to solve the problem.
    Reads input, performs calculations, and prints the output.
    """
    N, M, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # Case where the answer is 0.
    # If X_i = X_{i+1} for some i < M-1, one can show that any sequence
    # not containing X is also guaranteed not to contain a specific Y != X.
    # This contradicts the condition that X is the *only* missing subsequence.
    # The argument fails if the suffix X_{i+1...M} is homogeneous (all same elements),
    # as the constructed Y might equal X.
    for i in range(M - 1):
        if X[i] == X[i+1]:
            is_homogenous_suffix = True
            for j in range(i + 2, M):
                if X[j] != X[i]:
                    is_homogenous_suffix = False
                    break
            if not is_homogenous_suffix:
                print(0)
                return

    def count_avoiding_one(S):
        """Counts sequences of length N avoiding subsequence S."""
        m = len(S)
        dp = [[0] * (m + 1) for _ in range(N + 1)]
        dp[0][0] = 1

        # Precompute KMP failure function for S
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if S[i] == S[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        # Precompute transitions for the DP
        next_lps = [[0] * (K + 1) for _ in range(m + 1)]
        for j in range(m + 1):
            for char_code in range(1, K + 1):
                if j < m and char_code == S[j]:
                    next_lps[j][char_code] = j + 1
                else:
                    if j > 0:
                        next_lps[j][char_code] = next_lps[lps[j - 1]][char_code]
        
        for i in range(N):
            for j in range(m):
                if dp[i][j] == 0: continue
                for char_code in range(1, K + 1):
                    nj = next_lps[j][char_code]
                    if nj < m:
                        dp[i+1][nj] = (dp[i+1][nj] + dp[i][j]) % MOD
        
        res = 0
        for j in range(m):
            res = (res + dp[N][j]) % MOD
        return res

    def count_avoiding_two(S1, S2):
        """Counts sequences of length N avoiding subsequences S1 and S2."""
        m1, m2 = len(S1), len(S2)
        dp = [[0] * ((m1 + 1) * (m2 + 1)) for _ in range(N + 1)]
        dp[0][0] = 1

        # KMP precomputation for S1
        lps1 = [0] * m1
        l, i = 0, 1
        while i < m1:
            if S1[i] == S1[l]: l += 1; lps1[i] = l; i += 1
            else:
                if l != 0: l = lps1[l - 1]
                else: i += 1
        
        # KMP precomputation for S2
        lps2 = [0] * m2
        l, i = 0, 1
        while i < m2:
            if S2[i] == S2[l]: l += 1; lps2[i] = l; i += 1
            else:
                if l != 0: l = lps2[l - 1]
                else: i += 1

        # Precompute transitions
        next_lps_cache = {}
        def get_next_lps(j1, j2, char_code):
            if (j1, j2, char_code) in next_lps_cache:
                return next_lps_cache[(j1, j2, char_code)]
            
            # For S1
            nj1 = j1
            while nj1 > 0 and char_code != S1[nj1]: nj1 = lps1[nj1 - 1]
            if char_code == S1[nj1]: nj1 += 1
            
            # For S2
            nj2 = j2
            while nj2 > 0 and char_code != S2[nj2]: nj2 = lps2[nj2 - 1]
            if char_code == S2[nj2]: nj2 += 1
            
            next_lps_cache[(j1, j2, char_code)] = (nj1, nj2)
            return nj1, nj2

        for i in range(N):
            for state_idx in range(m1 * (m2 + 1)): # Only iterate over reachable states for S1
                if dp[i][state_idx] == 0: continue
                j1, j2 = state_idx // (m2 + 1), state_idx % (m2 + 1)
                
                for char_code in range(1, K + 1):
                    nj1, nj2 = get_next_lps(j1, j2, char_code)
                    if nj1 < m1 and nj2 < m2:
                        new_state_idx = nj1 * (m2 + 1) + nj2
                        dp[i + 1][new_state_idx] = (dp[i + 1][new_state_idx] + dp[i][state_idx]) % MOD
        
        res = 0
        for state_idx in range(m1 * m2):
            res = (res + dp[N][state_idx]) % MOD
        return res

    total_without_x = count_avoiding_one(X)
    
    if M > 1 and X[M - 1] != X[M - 2]:
        Y = list(X)
        Y[M - 2], Y[M - 1] = Y[M - 1], Y[M - 2]
        without_x_y = count_avoiding_two(X, Y)
        ans = (total_without_x - without_x_y + MOD) % MOD
        print(ans)
    else:
        print(total_without_x)

solve()