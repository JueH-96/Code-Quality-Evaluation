import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    MOD = 998244353

    # Compute prefix sums P[0..N]
    P = [0] * (N+1)
    for i in range(1, N+1):
        P[i] = (P[i-1] + A[i-1]) % MOD

    # Precompute inverses 1..K
    inv = [0] * (K+1)
    for i in range(1, K+1):
        inv[i] = pow(i, MOD-2, MOD)

    # Precompute binomial coefficients C(K, t)
    C = [0] * (K+1)
    C[0] = 1
    for t in range(1, K+1):
        # C[t] = C[t-1] * (K - t + 1) / t
        C[t] = C[t-1] * (K - t + 1) % MOD
        C[t] = C[t] * inv[t] % MOD

    # Combine with sign to get coef[t] = C(K,t) * (-1)^(K-t)
    coef = [0] * (K+1)
    for t in range(K+1):
        val = C[t]
        if ((K - t) & 1) == 1:
            val = -val
        coef[t] = val % MOD

    # Q[s] = sum of P[i]^s for i < current j
    Q = [0] * (K+1)

    ans = 0
    # Iterate j from 0 to N
    for j in range(N+1):
        # compute powers pows[t] = P[j]^t
        pows = [1] * (K+1)
        pj = P[j]
        for t in range(1, K+1):
            pows[t] = pows[t-1] * pj % MOD

        if j > 0:
            # contribution of all i<j to subarrays ending at j
            tmp = 0
            # sum over t: coef[t] * pows[t] * Q[K-t]
            for t in range(K+1):
                tmp += coef[t] * pows[t] * Q[K-t]
            ans = (ans + tmp) % MOD

        # now include P[j] into Q
        for s in range(K+1):
            Q[s] = (Q[s] + pows[s]) % MOD

    print(ans % MOD)

if __name__ == "__main__":
    main()