import sys

def solve():
    """
    This function implements the described algorithm to solve the problem.
    """
    # Fast I/O
    readline = sys.stdin.readline

    # Parse input
    try:
        N_str, M_str, K_str = readline().split()
        N, M, K = int(N_str), int(M_str), int(K_str)
        if N == 0:
            print(0)
            return
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # Handle empty input
        print(0)
        return

    MOD = 998244353
    # The maximum value of A_i is less than 2^20
    MAX_A_BITS = 20
    MAX_A_VAL = 1 << MAX_A_BITS

    # Step 1: Count frequencies of values in A
    v_A = [0] * MAX_A_VAL
    for x in A:
        v_A[x] += 1

    # Step 2: FWHT of the frequency map using integer arithmetic
    # hat_v_A[s] = sum_x (-1)^{<s,x>} v_A[x]
    # which is sum_{i=1 to N} (-1)^{<s,A_i>}
    hat_v_A = [int(x) for x in v_A]
    for i in range(MAX_A_BITS):
        for j in range(MAX_A_VAL):
            if (j >> i) & 1:
                u = hat_v_A[j ^ (1 << i)]
                v = hat_v_A[j]
                hat_v_A[j ^ (1 << i)] = u + v
                hat_v_A[j] = u - v

    # Step 3: Precompute S_k = M * sum_{i%M==0} [x^i] (1+x)^k (1-x)^(N-k)
    # This is done via DP: D(k,r) = D(k-1, r) + D(k-1, (r-1+M)%M)
    
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    # D_prev stores D[k-1], D_curr stores D[k]
    D_prev = [0] * M
    # Base case D[0][r] for P_0(x)=(1-x)^N
    for i in range(N + 1):
        term = nCr_mod(N, i)
        if i % 2 == 1:
            term = (MOD - term) % MOD
        D_prev[i % M] = (D_prev[i % M] + term) % MOD

    vals_S = [0] * (N + 1)
    vals_S[0] = (M * D_prev[0]) % MOD
    
    D_curr = [0] * M
    for k in range(1, N + 1):
        for r in range(M):
            D_curr[r] = (D_prev[r] + D_prev[(r - 1 + M) % M]) % MOD
        vals_S[k] = (M * D_curr[0]) % MOD
        D_prev, D_curr = D_curr, D_prev

    # Step 4: Construct hat_S_total
    hat_S_total = [0] * MAX_A_VAL
    for s in range(MAX_A_VAL):
        h = hat_v_A[s]
        # c0 = (N+h)/2, c1 = (N-h)/2
        # h and N have same parity, so N+h is even.
        c0 = (N + h) // 2
        hat_S_total[s] = vals_S[c0]

    # Step 5: Inverse FWHT (modulo MOD)
    S_total = hat_S_total
    for i in range(MAX_A_BITS):
        for j in range(MAX_A_VAL):
            if (j >> i) & 1:
                u = S_total[j ^ (1 << i)]
                v = S_total[j]
                S_total[j ^ (1 << i)] = (u + v) % MOD
                S_total[j] = (u - v + MOD) % MOD
    
    # Step 6: Scale and calculate the final answer
    inv_U = pow(MAX_A_VAL, MOD - 2, MOD)
    inv_M = pow(M, MOD - 2, MOD)
    
    ans = 0
    
    # F(X) is the number of non-empty subsequences with length multiple of M and XOR sum X
    # F(X) = S_0[X] for X>0, and F(0) = S_0[0]-1
    # S_0[X] = S_total[X] * inv_U * inv_M
    
    # X=0 case
    count_0 = (S_total[0] * inv_U % MOD * inv_M % MOD - 1 + MOD) % MOD
    if K == 0:
        term = 1
    else: # K > 0
        term = 0 # pow(0, K, MOD) is 0
    ans = (ans + count_0 * term) % MOD
    
    # X > 0 cases
    for i in range(1, MAX_A_VAL):
        count_i = (S_total[i] * inv_U % MOD * inv_M % MOD) % MOD
        if count_i > 0:
            term = pow(i, K, MOD)
            ans = (ans + count_i * term) % MOD
            
    print(ans)

if __name__ == "__main__":
    solve()