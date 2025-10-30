import sys

def solve():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()

    MOD = 998244353

    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    # pref_W[k] = count of 'W' in S_str[0...k-1] (i.e., among vertices 1...k)
    # pref_B[k] = count of 'B' in S_str[0...k-1] (i.e., among vertices 1...k)
    pref_W = [0] * (2 * N + 1)
    pref_B = [0] * (2 * N + 1)

    for i in range(2 * N):
        pref_W[i+1] = pref_W[i]
        pref_B[i+1] = pref_B[i]
        if S_str[i] == 'W':
            pref_W[i+1] += 1
        else:
            pref_B[i+1] += 1

    # Check for trivial zero cases.
    # A "cut" is after vertex k_vtx. V1 = {1,...,k_vtx}, V2 = {k_vtx+1,...,2N}.
    # k_vtx ranges from 1 to 2N-1.
    for k_vtx in range(1, 2 * N):
        # If all N white vertices are in V1={1,...,k_vtx}, i.e., pref_W[k_vtx] == N.
        # Then any white vertex w satisfies w <= k_vtx.
        # No added edge (w,b) can have w > k_vtx.
        # So, the condition "exists edge (w,b) with w > k_vtx and b <= k_vtx" is false.
        # Graph is not strongly connected.
        if pref_W[k_vtx] == N:
            print(0)
            return
        
        # If all N black vertices are in V2={k_vtx+1,...,2N}, i.e., pref_B[k_vtx] == 0.
        # (Meaning no black vertices in V1).
        # Then any black vertex b satisfies b > k_vtx.
        # No added edge (w,b) can have b <= k_vtx.
        # So, the condition "exists edge (w,b) with w > k_vtx and b <= k_vtx" is false.
        # Graph is not strongly connected.
        if pref_B[k_vtx] == 0:
            print(0)
            return
    
    # dp[m] = number of ways to pair vertices {1,...,2m} (using S_str[0...2m-1])
    # such that this configuration is "indecomposable".
    dp = [0] * (N + 1)
    
    for m in range(1, N + 1):
        # An indecomposable block of 2m vertices must itself be balanced:
        # m white vertices and m black vertices.
        # pref_W[2*m] is count of 'W' in S_str[0...2m-1].
        if pref_W[2*m] != m: # This also implies pref_B[2*m] != m
            dp[m] = 0 
            continue

        # Total ways to pair m Ws and m Bs is m!
        current_val = fact[m]
        
        # Subtract ways where {1,...,2m} is decomposable.
        # This happens if for some j < m, {1,...,2j} is an indecomposable block (dp[j] ways)
        # and the remaining {2j+1,...,2m} are paired arbitrarily ((m-j)! ways).
        for j in range(1, m): 
            # The prefix S_str[0...2j-1] (vertices {1,...,2j}) must be balanced
            # to be the first part of a decomposition.
            if pref_W[2*j] == j: # This also implies pref_B[2*j] == j
                term = (dp[j] * fact[m-j]) % MOD
                current_val = (current_val - term + MOD) % MOD
        dp[m] = current_val
        
    print(dp[N])

solve()