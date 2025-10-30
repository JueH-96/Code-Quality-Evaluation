import sys

def main():
    """
    Main function to solve the problem.
    """
    # Read input from stdin
    try:
        line = sys.stdin.readline().strip()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    MOD = 998244353

    # Step 1: Identify primes <= M. Since M <= 16, we can hardcode them.
    all_possible_primes = [2, 3, 5, 7, 11, 13]
    primes = [p for p in all_possible_primes if p <= M]
    r = len(primes)
    D = 1 << r

    # Helper function to get the exponent of a prime in a number's factorization.
    def get_v(p, n):
        count = 0
        while n > 0 and n % p == 0:
            count += 1
            n //= p
        return count

    # Step 2: Precompute v_p(x) for x in {1..M} and p in primes.
    v_table = [[0] * r for _ in range(M + 1)]
    for x in range(1, M + 1):
        for i, p in enumerate(primes):
            v_table[x][i] = get_v(p, x)

    # Step 3: Precompute C(L) for each subset L of primes.
    # C(L) = sum_{x=1..M} product_{p in L} v_p(x)
    C = [0] * D
    for mask in range(D):
        val = 0
        for x in range(1, M + 1):
            term = 1
            for i in range(r):
                if (mask >> i) & 1:
                    term *= v_table[x][i]
            val += term
        C[mask] = val % MOD

    # Step 4: Build the transition matrix T.
    # T[j][k] = C(j \ k) if k is a submask of j.
    T = [[0] * D for _ in range(D)]
    for j_mask in range(D):
        for k_mask in range(D):
            if (k_mask & j_mask) == k_mask:
                diff_mask = j_mask ^ k_mask
                T[j_mask][k_mask] = C[diff_mask]

    # Helper functions for matrix operations.
    def mat_mul(A, B, size, mod):
        res = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    res[i][j] = (res[i][j] + A[i][k] * B[k][j]) % mod
        return res

    def mat_pow(A, n, size, mod):
        res = [[0] * size for _ in range(size)]
        for i in range(size):
            res[i][i] = 1
        base = A
        while n > 0:
            if n % 2 == 1:
                res = mat_mul(res, base, size, mod)
            base = mat_mul(base, base, size, mod)
            n //= 2
        return res

    # Step 5: Use an augmented matrix to find the sum of matrix powers.
    # We need Sum(T^k) for k=1..N. This can be found by computing A^N where A = [[T, 0], [T, I]].
    size_aug = 2 * D
    A_aug = [[0] * size_aug for _ in range(size_aug)]

    for i in range(D):
        for j in range(D):
            A_aug[i][j] = T[i][j]
            A_aug[i + D][j] = T[i][j]
    for i in range(D):
        A_aug[i + D][i + D] = 1
    
    if N == 0:
        print(0)
        return

    A_pow_N = mat_pow(A_aug, N, size_aug, MOD)
    
    # Step 6: Calculate the final sum.
    # The total score is the sum of all elements in the vector (Sum(T^k) * X_0),
    # where X_0 corresponds to the empty sequence. This vector is the first
    # column of the sum-of-powers matrix, located in the bottom-left block of A_pow_N.
    
    total_score = 0
    for i in range(D):
        total_score = (total_score + A_pow_N[i + D][0]) % MOD
    
    print(total_score)

if __name__ == "__main__":
    main()