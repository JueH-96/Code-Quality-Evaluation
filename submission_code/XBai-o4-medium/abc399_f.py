import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    # Precompute combinatorial numbers
    max_n = K
    comb = [[0]*(max_n+2) for _ in range(max_n+2)]
    for n in range(max_n+1):
        comb[n][0] = 1
        comb[n][n] = 1
        for k in range(1, n):
            comb[n][k] = comb[n-1][k] + comb[n-1][k-1]
    
    prev = [0]*(K+1)
    ans = 0
    for a in A:
        # Compute powers of a
        pow_a = [1]*(K+1)
        for e in range(1, K+1):
            pow_a[e] = (pow_a[e-1] * a) % MOD
        curr = [0]*(K+1)
        for m in range(K+1):
            total = 0
            for t in range(m+1):
                c = comb[m][t]
                exponent = m - t
                term = c * pow_a[exponent] % MOD
                term = term * prev[t] % MOD
                total = (total + term) % MOD
            add_term = pow_a[m] % MOD
            curr[m] = (total + add_term) % MOD
        ans = (ans + curr[K]) % MOD
        prev = curr.copy()
    print(ans % MOD)

if __name__ == '__main__':
    main()