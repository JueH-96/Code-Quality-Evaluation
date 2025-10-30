MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    B = list(map(int, input[idx:idx+N]))
    idx += N
    
    # Precompute prefix sums of -1 counts
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + (B[i] == -1)
    
    # Precompute pow_table[x][a] = x^a mod MOD for x up to M and a up to 2000
    max_a = 2000
    pow_table = [[1]*(max_a + 1) for _ in range(M + 1)]
    for x in range(M + 1):
        for a in range(1, max_a + 1):
            pow_table[x][a] = (pow_table[x][a-1] * x) % MOD
    
    total = 0
    
    for j in range(1, N + 1):
        current_B = B[j-1]
        if current_B != -1:
            if current_B == M:
                continue
            # Check all fixed B_i (i < j) are > current_B
            valid = True
            for i in range(j-1):
                if B[i] != -1 and B[i] <= current_B:
                    valid = False
                    break
            if not valid:
                continue
            a = prefix[j-1]
            term = pow(M - current_B, a, MOD)
            remaining = pow(M, prefix[N] - prefix[j], MOD)
            contribution = (term * remaining) % MOD
            total = (total + contribution) % MOD
        else:
            # Compute min_fixed for i < j
            fixed_values = [B[i] for i in range(j-1) if B[i] != -1]
            if not fixed_values:
                K = M
            else:
                min_fixed = min(fixed_values)
                if min_fixed <= 1:
                    K = 0
                else:
                    K = min_fixed - 1
            a = prefix[j-1]
            sum_k = 0
            if K > 0:
                start = M - K
                end = M - 1
                if start < 0:
                    start = 0
                for x in range(start, end + 1):
                    sum_k = (sum_k + pow_table[x][a]) % MOD
            remaining = pow(M, prefix[N] - prefix[j], MOD)
            contribution = (sum_k * remaining) % MOD
            total = (total + contribution) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()