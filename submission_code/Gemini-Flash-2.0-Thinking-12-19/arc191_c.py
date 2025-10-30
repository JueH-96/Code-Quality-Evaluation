import sys

# The function to solve a single test case
def solve():
    # Read the input integer N
    N = int(sys.stdin.readline())

    # We need to find a pair of positive integers (A, M) such that
    # A^n - 1 is a multiple of M, and the smallest such positive integer n is N.
    # This means A^n = 1 (mod M), and the smallest positive n satisfying this is N.
    # This is the definition of the multiplicative order of A modulo M, ord_M(A) = N,
    # typically defined when gcd(A, M) = 1. The problem statement guarantees a solution exists,
    # implying that A^n = 1 (mod M) must be achievable for some positive n.

    # We use the construction (A, M) = (N + 1, N^2).
    # Let's verify this construction.
    # A = N + 1
    # M = N * N

    # Check constraints: 1 <= N <= 10^9, 1 <= A, M <= 10^18
    # N is a positive integer.
    # A = N + 1 is positive. Minimum A is 1+1 = 2 (when N=1). Maximum A is 10^9 + 1, which fits within 10^18.
    # M = N * N is positive. Minimum M is 1*1 = 1 (when N=1). Maximum M is (10^9)^2 = 10^18, which fits within 10^18.
    # All constraint conditions on A and M are met.

    # Verify that ord_{N^2}(N+1) = N.
    # First, check gcd(A, M) = gcd(N+1, N^2).
    # Using the property gcd(a, b) = gcd(a - kb, b), we have gcd(N+1, N^2) = gcd(N+1 - N * 1, N^2) = gcd(1, N^2) = 1.
    # So, gcd(N+1, N^2) = 1 for all N >= 1. The standard definition of multiplicative order applies.

    # We need to show that (N+1)^N = 1 (mod N^2).
    # Using the binomial theorem: (N+1)^N = sum_{k=0 to N} binom(N, k) * N^k
    # (N+1)^N = binom(N, 0)*N^0 + binom(N, 1)*N^1 + binom(N, 2)*N^2 + binom(N, 3)*N^3 + ... + binom(N, N)*N^N
    # (N+1)^N = 1 * 1 + N * N + (N * (N-1) // 2) * N^2 + ... + 1 * N^N
    # (N+1)^N = 1 + N^2 + (N(N-1)//2)*N^2 + ... + N^N
    # Considering this equation modulo N^2:
    # All terms from binom(N, 2)*N^2 onwards are divisible by N^2.
    # So, (N+1)^N = 1 + N^2 + 0 + ... + 0 (mod N^2)
    # (N+1)^N = 1 + N^2 (mod N^2)
    # (N+1)^N = 1 (mod N^2) since N^2 is a multiple of N^2.
    # This shows that A^N - 1 is a multiple of M.

    # Next, we need to show that (N+1)^k != 1 (mod N^2) for 1 <= k < N.
    # Using the binomial theorem: (N+1)^k = sum_{j=0 to k} binom(k, j) * N^j
    # (N+1)^k = binom(k, 0)*N^0 + binom(k, 1)*N^1 + binom(k, 2)*N^2 + ... + binom(k, k)*N^k
    # (N+1)^k = 1 * 1 + k * N + (k * (k-1) // 2) * N^2 + ... + N^k
    # Considering this equation modulo N^2:
    # All terms from binom(k, 2)*N^2 onwards are divisible by N^2, provided k >= 2.
    # If k = 1: (N+1)^1 = 1 + 1*N = 1 + N. Modulo N^2, 1+N != 1 if N^2 > 1. N^2 = 1 only if N=1.
    # If N=1, the range 1 <= k < N is empty, so this condition is vacuously true.
    # If N > 1, then 1+N != 1 (mod N^2) since N is not a multiple of N^2.
    # If k >= 2: (N+1)^k = 1 + k*N + terms_divisible_by_N^2 (mod N^2)
    # (N+1)^k = 1 + k*N (mod N^2).
    # We need to show 1 + k*N != 1 (mod N^2) for 1 <= k < N.
    # This is equivalent to showing k*N != 0 (mod N^2) for 1 <= k < N.
    # k*N is a multiple of N. For k*N to be a multiple of N^2, k must be a multiple of N.
    # However, the range 1 <= k < N contains no integers k that are multiples of N.
    # Thus, k*N is not a multiple of N^2 for 1 <= k < N.
    # So, 1 + k*N != 1 (mod N^2) for 1 <= k < N.
    # This shows that (N+1)^k != 1 (mod N^2) for 1 <= k < N.

    # Combining the two conditions, the smallest positive integer n such that (N+1)^n = 1 (mod N^2) is exactly N.
    # The pair (A, M) = (N+1, N^2) is a valid solution.

    A = N + 1
    M = N * N

    # Print the output A and M, separated by a space
    print(A, M)

# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case by calling the solve function
for _ in range(T):
    solve()