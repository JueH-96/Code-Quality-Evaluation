import sys

# It is recommended to use fast I/O for large inputs
# For example, by reading the entire input at once
# input = sys.stdin.read
# Then process the read string
# But for typical competitive programming platforms, this is fine.
sys.setrecursionlimit(4000)

def solve():
    """
    This function encapsulates the entire solution logic.
    """
    try:
        # Read from stdin
        stdin_lines = sys.stdin.readlines()
        if not stdin_lines:
            return
        N = int(stdin_lines[0])
        A = list(map(int, stdin_lines[1].split()))
        B = list(map(int, stdin_lines[2].split()))
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors gracefully
        return

    MOD = 998244353

    # Step 1: Determine x_i and relabel
    fA = [(val + 1) // 2 for val in A]
    
    counts = [0] * (N + 1)
    for val in fA:
        if not (1 <= val <= N):
            print(0)
            return
        counts[val] += 1
    
    if any(c != 1 for c in counts[1:]):
        print(0)
        return

    # Relabeling map: original index i -> new index x_i-1
    new_A = [0] * N
    new_B = [0] * N
    for i in range(N):
        new_idx = fA[i] - 1
        new_A[new_idx] = A[i]
        new_B[new_idx] = B[i]
    
    A = new_A
    B = new_B

    # Step 2: Determine z_i constraints
    z = [-1] * N
    I_unknown_indices = []
    
    Z_known_vals = set()
    for i in range(N):
        if B[i] != -1:
            z_val = (B[i] + 1) // 2
            if not (1 <= z_val <= N) or z_val in Z_known_vals:
                print(0)
                return
            if z_val == i + 1:
                print(0)
                return
            z[i] = z_val
            Z_known_vals.add(z_val)
        else:
            I_unknown_indices.append(i)

    Z_free_vals = sorted(list(set(range(1, N + 1)) - Z_known_vals))
    
    if len(I_unknown_indices) != len(Z_free_vals):
        print(0)
        return

    # Step 3: Count valid z permutations using DP and Inclusion-Exclusion
    S0 = {i for i in range(N) if A[i] % 2 == 1}
    
    U0 = [i for i in I_unknown_indices if i in S0]
    U1 = [i for i in I_unknown_indices if i not in S0]
    
    n0, n1 = len(U0), len(U1)
    
    # DP for counting permutations where elements of one color map to another
    dp = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    dp[0][0] = 1
    for i in range(n0 + 1):
        for j in range(n1 + 1):
            if i > 0:
                dp[i][j] = (dp[i][j] + j * dp[i-1][j]) % MOD
            if j > 0:
                dp[i][j] = (dp[i][j] + i * dp[i][j-1]) % MOD
    
    # Derangement part via Inclusion-Exclusion
    fact = [1] * (N + 1)
    invfact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invfact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1):
        invfact[i] = (invfact[i + 1] * (i + 1)) % MOD

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (invfact[r] * invfact[n - r]) % MOD
        return (num * den) % MOD

    Z_free_set = set(Z_free_vals)
    
    # Forbidden self-mappings within unknown sets
    u0_z_self = [i for i in U0 if i + 1 in Z_free_set]
    u1_z_self = [i for i in U1 if i + 1 in Z_free_set]
    
    len_u0_self = len(u0_z_self)
    len_u1_self = len(u1_z_self)

    ans = 0
    for i in range(len_u0_self + 1):
        for j in range(len_u1_self + 1):
            rem_n0 = n0 - i
            rem_n1 = n1 - j
            
            if rem_n0 < 0 or rem_n1 < 0:
                continue
            
            term = dp[rem_n0][rem_n1]
            term = (term * nCr_mod(len_u0_self, i)) % MOD
            term = (term * nCr_mod(len_u1_self, j)) % MOD
            
            if (i + j) % 2 == 1:
                ans = (ans - term + MOD) % MOD
            else:
                ans = (ans + term) % MOD
    
    # Check for contradictions in the fixed part of z
    visited = [False] * (N + 1)
    for i in range(N):
        if z[i] != -1 and not visited[i+1]:
            path = []
            curr = i + 1
            while curr != -1 and not visited[curr]:
                visited[curr] = True
                path.append(curr-1)
                if z[curr-1] != -1:
                    curr = z[curr-1]
                else:
                    curr = -1

            if curr != -1: # Cycle detected
                try:
                    cycle_start_idx = path.index(curr-1)
                    cycle = path[cycle_start_idx:]
                    is_s0 = all(c in S0 for c in cycle)
                    if is_s0:
                        print(0)
                        return
                    is_s1 = all(c not in S0 for c in cycle)
                    if is_s1:
                        print(0)
                        return
                except ValueError:
                    # Not a cycle within this component, but might be part of a larger one
                    # This check is sufficient for cycles completely within the known part
                    pass

    print(ans)

if __name__ == "__main__":
    solve()