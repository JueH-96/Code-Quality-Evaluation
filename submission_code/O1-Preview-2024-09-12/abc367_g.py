# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    from collections import defaultdict

    # First, compute counts[0..M-1]: counts of subsequences with length mod M == r
    M_counts = [0] * M
    # Compute (1 + y) ** N mod y^M - 1
    # We can use exponentiation over polynomials modulo y^M -1

    # Represent polynomials as lists, P(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...

    def poly_mul(a, b):
        M = len(a)
        res = [0] * M
        for i in range(M):
            for j in range(M):
                res[(i + j) % M] = (res[(i + j) % M] + a[i] * b[j]) % (MOD)
        return res

    def poly_pow(a, n):
        res = [0] * len(a)
        res[0] = 1
        while n > 0:
            if n % 2 == 1:
                res = poly_mul(res, a)
            a = poly_mul(a, a)
            n //= 1 << 1
        return res

    # Initialize P(y) = 1 + y
    P = [1] * M
    P[1 % M] += 1  # P = [1,1,0,...,0]

    counts = [0] * M
    # Compute (1 + y)^N mod y^M -1
    # Since a = P, n = N, M is mod
    def poly_pow(a, n):
        res = [0] * len(a)
        res[0] = 1
        while n > 0:
            if n % 2 == 1:
                res = poly_mul(res, a)
            a = poly_mul(a, a)
            n >>=1
        return res

    counts = poly_pow(P, N)

    total_subseq = pow(2, N, MOD)
    inv_total_subseq = pow(total_subseq, MOD - 2, MOD)
    counts_0 = counts[0]

    # Now, perform DP using dictionaries
    F = defaultdict(int)
    F[0, 0] = 1  # (xor_value, length mod M) -> count

    for a in A:
        Fnew = F.copy()
        for (x, r), cnt in F.items():
            x_new = x ^ a
            r_new = (r + 1) % M
            key = (x_new, r_new)
            Fnew[key] = (Fnew.get(key, 0) + cnt) % MOD
        F = Fnew

    # Now, sum over all (x, r), where r == 0
    ans = 0
    for (x, r), cnt in F.items():
        if r == 0 and cnt > 0:
            if (x == 0 and cnt == 1):
                continue  # Exclude empty subsequence
            # Multiply cnt * (x ^ K) % MOD
            ans = (ans + cnt * pow(x, K, MOD)) % MOD
    print(ans)
    
threading.Thread(target=main).start()