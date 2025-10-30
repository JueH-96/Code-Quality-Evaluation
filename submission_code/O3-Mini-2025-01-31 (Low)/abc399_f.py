def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # Compute prefix sums: P[0] = 0, P[i] = A_1 + ... + A_i (mod mod)
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = (P[i-1] + A[i-1]) % mod
        
    # We need to compute:
    # Sum_{1 <= l <= r <= N} (sum_{i=l}^r A_i)^K.
    # Notice that for a subarray l...r, sum = P[r] - P[l-1]. Changing index:
    # Let i = l-1 and j = r, so 0 <= i < j <= N.
    # Then we need to compute: sum_{0 <= i < j <= N} (P[j] - P[i])^K.
    #
    # Expanding using the binomial theorem:
    #   (P[j] - P[i])^K = sum_{a=0}^{K} binom(K, a) * (-1)^(K-a) * P[j]^a * P[i]^(K-a)
    #
    # We can rearrange the sum:
    #   Answer = sum_{j=1}^{N} sum_{i=0}^{j-1} sum_{a=0}^{K} binom(K, a) * (-1)^(K-a) *
    #            P[j]^a * P[i]^(K-a)
    #         = sum_{j=1}^{N} sum_{a=0}^{K} binom(K, a) * (-1)^(K-a) *
    #            (P[j]^a) * (sum_{i=0}^{j-1} P[i]^(K-a))
    #
    # Since K is at most 10, we can precompute for each exponent value with O(N*K).
    
    # Precompute powers: for each exponent a (0 <= a <= K), let 
    #   pow_arr[a][i] = P[i]^a modulo mod.
    pow_arr = [[0] * (N + 1) for _ in range(K+1)]
    for i in range(N+1):
        pow_arr[0][i] = 1  # any number^0 == 1
    for a in range(1, K+1):
        for i in range(N+1):
            pow_arr[a][i] = (pow_arr[a-1][i] * P[i]) % mod

    # Precompute cumulative sums for each exponent:
    #   cum[a][j] = sum_{i=0}^{j} pow_arr[a][i] modulo mod.
    cum = [[0] * (N + 1) for _ in range(K+1)]
    for a in range(K+1):
        cum[a][0] = pow_arr[a][0]
        for j in range(1, N+1):
            cum[a][j] = (cum[a][j-1] + pow_arr[a][j]) % mod
            
    # Precompute binomial coefficients for 0 <= a <= K.
    binom = [[0]*(K+1) for _ in range(K+1)]
    for i in range(K+1):
        binom[i][0] = 1
        for j in range(1, i+1):
            binom[i][j] = (binom[i-1][j-1] + binom[i-1][j]) % mod

    ans = 0
    # Loop over j (the ending index in prefix sums, corresponding to r) and for each exponent split.
    for j in range(1, N+1):
        for a in range(0, K+1):
            # Coefficient: binom(K, a) * (-1)^(K-a)
            # (-1) mod mod is (mod-1), and power parity gives (-1)^(K-a)
            sign = 1 if ((K - a) & 1) == 0 else (mod - 1)
            coeff = binom[K][a]
            # Multiply with P[j]^a
            term = coeff * sign % mod
            term = term * pow_arr[a][j] % mod
            # The sum over i from 0 to j-1 for the remaining exponent (K-a):
            term = term * (cum[K-a][j-1] if j-1 >= 0 else 0) % mod
            ans = (ans + term) % mod
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()