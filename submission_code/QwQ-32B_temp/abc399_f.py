import sys
MOD = 998244353

def binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Compute prefix sums S
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = (S[i-1] + A[i-1]) % MOD
    
    # Precompute coefficients for each i
    coeffs = []
    for i in range(K + 1):
        c = binom(K, i)
        sign = (-1) ** (K - i)
        coeff = (c * sign) % MOD
        coeffs.append(coeff)
    
    # Precompute prefix arrays for each d from 0 to K
    prefixes = []
    for d in range(K + 1):
        pre = [0] * (N + 1)
        pre[0] = pow(S[0], d, MOD)
        for m in range(1, N + 1):
            pre[m] = (pre[m-1] + pow(S[m], d, MOD)) % MOD
        prefixes.append(pre)
    
    total = 0
    for r in range(1, N + 1):
        Sr = S[r]
        contrib = 0
        for i in range(K + 1):
            d = K - i
            coeff = coeffs[i]
            sum_part = prefixes[d][r-1]
            term1 = pow(Sr, i, MOD)
            term = coeff * term1
            term %= MOD
            term = term * sum_part
            term %= MOD
            contrib = (contrib + term) % MOD
        total = (total + contrib) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()