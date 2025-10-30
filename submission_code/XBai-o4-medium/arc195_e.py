import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    A = [0] * (N + 2)  # A[2..N]
    for i in range(2, N + 1):
        A[i] = int(input[ptr])
        ptr += 1
    
    # Precompute inverses
    inv = [1] * (N + 2)
    for i in range(2, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Precompute prefix sums S and T
    S = [0] * (N + 2)
    T = [0] * (N + 2)
    for i in range(2, N + 1):
        term_S = A[i] * inv[i] % MOD
        S[i] = (S[i-1] + term_S) % MOD
        term_T = term_S * inv[i] % MOD
        T[i] = (T[i-1] + term_T) % MOD
    
    # Precompute (N-1)!
    fact = 1
    for i in range(2, N):  # i runs from 2 to N-1 - 1
        fact = fact * i % MOD
    
    # Process queries
    for _ in range(Q):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        
        # Compute sum_pu
        if u >= 2:
            sum_pu = (A[u] + S[u-1]) % MOD
        else:
            sum_pu = 0
        
        # Compute sum_pv
        if v >= 2:
            sum_pv = (A[v] + S[v-1]) % MOD
        else:
            sum_pv = 0
        
        # Compute sum_pu_pv
        if u >= 2:
            sum_pu_pv_part1 = T[u-1]
            sum_pu_pv_part2 = A[u] * inv[u] % MOD
        else:
            sum_pu_pv_part1 = 0
            sum_pu_pv_part2 = 0
        sum_pu_pv = (sum_pu_pv_part1 + sum_pu_pv_part2) % MOD
        
        total = (sum_pu + sum_pv - 2 * sum_pu_pv) % MOD
        if total < 0:
            total += MOD
        
        ans = (total * fact) % MOD
        print(ans)
        
if __name__ == "__main__":
    main()