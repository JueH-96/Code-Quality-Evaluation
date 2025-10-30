import sys

def solve():
    N = int(sys.stdin.readline())

    # We need to find A, M such that:
    # 1. 1 <= A, M <= 10^18
    # 2. A^n - 1 is a multiple of M, and the smallest such n is N.
    # This means the multiplicative order of A modulo M is N (ord_M(A) = N).

    # Let's choose A = N + 1 and M = N^2.

    # Check condition 1: A and M within 10^18.
    # N is given between 1 and 10^9.
    # A = N + 1: Max A = 10^9 + 1, which is much less than 10^18.
    # M = N^2: Max M = (10^9)^2 = 10^18. This is within the limit.
    # So, 1 <= A, M <= 10^18 is satisfied.

    # Check condition 2: ord_M(A) = N.
    # We need to show that (N+1)^n - 1 is a multiple of N^2 if and only if n is a multiple of N,
    # and therefore the smallest positive such n is N.

    # Using binomial expansion for (N+1)^k - 1:
    # (N+1)^k - 1 = sum_{j=0 to k} (k choose j) * N^j - 1
    #               = (k choose 0)*N^0 + (k choose 1)*N^1 + (k choose 2)*N^2 + ... + (k choose k)*N^k - 1
    #               = 1 + k*N + (k*(k-1)/2)*N^2 + ... + N^k - 1
    #               = k*N + (terms divisible by N^2)
    # So, (N+1)^k - 1 is congruent to k*N (mod N^2).

    # For (N+1)^k - 1 to be a multiple of N^2, we need k*N to be congruent to 0 (mod N^2).
    # This means N^2 must divide k*N.
    # Dividing by N (since N is positive), we get that N must divide k.

    # Therefore, the smallest positive integer k for which (N+1)^k - 1 is a multiple of N^2
    # is the smallest positive multiple of N, which is N itself.

    # This proves that for A = N+1 and M = N^2, the smallest n is indeed N.

    # Example for N=1:
    # A = 1 + 1 = 2
    # M = 1*1 = 1
    # For n=1, 2^1 - 1 = 1, which is a multiple of M=1.
    # For any n > 1, 2^n - 1 will also be a multiple of 1.
    # The smallest n is 1, which matches N.

    A = N + 1
    M = N * N

    sys.stdout.write(f"{A} {M}
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()