import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    mod = 998244353

    # Precompute binomial coefficients C[K][d] for d=0..K by Pascal's triangle
    C = [0] * (K+1)
    # We'll generate the K-th row of Pascal
    row = [1]
    for i in range(1, K+1):
        newrow = [1] * (i+1)
        for j in range(1, i):
            newrow[j] = (row[j-1] + row[j]) % mod
        row = newrow
    # row now holds C(K,d)
    for d in range(K+1):
        C[d] = row[d]

    # Precompute weights W[d] = C(K,d) * (-1)^(K-d) mod
    W = [0] * (K+1)
    for d in range(K+1):
        sign = -1 if ((K-d) & 1) else 1
        W[d] = C[d] * sign
        W[d] %= mod

    # We'll keep prefix sums S[m] = sum_{i<j} P_i^m
    # Initialize for j=0: P_0 = 0, so 0^0 = 1 contributes to S[0]
    S = [0] * (K+1)
    S[0] = 1

    ans = 0
    prefix = 0
    for x in A:
        prefix = (prefix + x) % mod
        # compute prefix^m for m=0..K
        pp = [1] * (K+1)
        for m in range(1, K+1):
            pp[m] = pp[m-1] * prefix % mod

        # accumulate contribution for subarrays ending here
        total = 0
        # sum over d: W[d] * prefix^d * S[K-d]
        for d in range(K+1):
            total += W[d] * pp[d] * S[K-d]
        total %= mod
        ans = (ans + total) % mod

        # now include this prefix in S for future j
        for m in range(K+1):
            S[m] = (S[m] + pp[m]) % mod

    print(ans % mod)

if __name__ == "__main__":
    main()