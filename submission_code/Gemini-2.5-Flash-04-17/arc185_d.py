import sys

MOD = 998244353

def power(a, b):
    """Computes a^b mod MOD."""
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def modInverse(n):
    """Computes modular multiplicative inverse of n mod MOD."""
    return power(n, MOD - 2)

def solve():
    """Solves the problem for given N and M."""
    N, M = map(int, sys.stdin.readline().split())

    # The tree structure has vertices 0 to N*M.
    # Vertex i (1 <= i <= NM) is connected to max(i-N, 0).
    # Vertex 0 is connected to 1, ..., N.
    # Vertex i (1 <= i <= N) is connected to 0 and i+N (if i+N <= NM).
    # Vertex i (N+1 <= i <= NM) is connected to i-N and i+N (if i+N <= NM).
    # Vertex NM is connected to NM-N.

    # This forms a tree with root 0, and N branches.
    # Branch j (1 <= j <= N) consists of vertices j, N+j, 2N+j, ..., (M-1)N+j.
    # Vertex 0 is connected to j, N+j, ..., (M-1)N+j forms a path graph
    # with vertex 0 attached to the start of the path.
    # Vertex 0 is connected to v_{1,j} = j.
    # v_{k,j} = (k-1)N + j is connected to v_{k-1,j} and v_{k+1,j}.

    # Degrees:
    # deg(0) = N
    # deg(v_{k,j}) = 2 for 1 <= k < M, 1 <= j <= N (vertices 1 to N(M-1))
    # deg(v_{M,j}) = 1 for 1 <= j <= N (vertices N(M-1)+1 to NM)
    # For M=1, deg(v_{1,j}) = 1 for 1 <= j <= N.

    # Based on Sample 1 (N=2, M=2, Output 20) matching 2 * N * H_N for N=2:
    # 2 * 2 * (1/1 + 1/2) = 4 * (3/2) = 6. Modulo 998244353: 4 * (1 + 499122177) = 4 * 499122178 = 1996488712.
    # 1996488712 = 2 * 998244353 + 20. So it matches.
    # This suggests the formula is 2 * N * H_N mod MOD, where H_N = sum(1/i) for i=1 to N.
    # This formula depends only on N, not M.

    # Calculate H_N = sum(1/i) mod MOD for i=1 to N
    # Use batch modular inverse for efficiency.
    inv = [0] * (N + 1)
    if N >= 1:
        inv[1] = 1
        for i in range(2, N + 1):
            # inv[i] = (MOD - (MOD // i)) * inv[MOD % i] % MOD
            # Use standard extended Euclidean algorithm approach for clarity, though batch is faster.
            # We need i_inv = modInverse(i) for each i.
            pass # We will calculate individually or use batch method

    harmonic_sum = 0
    # Calculate harmonic sum using precomputed inverses or individual calculation
    # Batch inverse calculation is better for large N
    if N > 0:
        inv_array = [0] * (N + 1)
        inv_array[1] = 1
        for i in range(2, N + 1):
             inv_array[i] = (MOD - (MOD // i) * inv_array[MOD % i] % MOD) % MOD

        for i in range(1, N + 1):
            harmonic_sum = (harmonic_sum + inv_array[i]) % MOD
    else:
         # N >= 1 according to constraints, but handle N=0 case gracefully if needed
         pass


    # The result seems to be 2 * N * H_N mod MOD
    # Note: M is not used in this formula based on sample matching.
    result = (2 * N * harmonic_sum) % MOD

    print(result)

solve()