def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353

    K = int(input_data[0])
    C = list(map(int, input_data[1:]))

    # Precompute factorials and inverse factorials for 0..K
    fact = [1] * (K+1)
    invfact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % MOD
    invfact[K] = pow(fact[K], MOD - 2, MOD)  # Fermat's little theorem for inverse
    for i in reversed(range(K)):
        invfact[i] = invfact[i+1] * (i+1) % MOD

    # Precompute binomial coefficients (Pascal's triangle) for 0..K
    binom = [[0]*(K+1) for _ in range(K+1)]
    for n in range(K+1):
        binom[n][0] = 1
        for r in range(1, n+1):
            binom[n][r] = (binom[n-1][r-1] + binom[n-1][r]) % MOD

    # invfactC[j] = 1/j! for 0..K
    invfactC = [invfact[j] for j in range(K+1)]

    # G[n] will hold the coefficient of x^n / n! in the EGF product so far
    G = [0] * (K+1)
    G[0] = 1

    # Multiply EGF factors for each of the 26 letters
    for i in range(26):
        max_use = min(K, C[i])
        # Construct F for letter i: F[j] = 1/j! for j=0..max_use, else 0
        F = [0]*(K+1)
        for j in range(max_use+1):
            F[j] = invfactC[j]

        # EGF multiply: newG[n] = sum_{i=0..n} binom[n][i] * G[i] * F[n-i]
        newG = [0]*(K+1)
        for n in range(K+1):
            s = 0
            bn = binom[n]
            for i2 in range(n+1):
                s += bn[i2] * G[i2] * F[n-i2]
            newG[n] = s % MOD
        G = newG

    # Sum up G[n] for n=1..K
    ans = sum(G[1:K+1]) % MOD
    print(ans)

# Call main() to execute solution
if __name__ == "__main__":
    main()