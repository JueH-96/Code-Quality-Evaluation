def solve():
    N = int(input())

    # We need to find (A, M) such that ord_M(A) = N.
    # A and M are positive integers <= 10^18.
    # N is a positive integer <= 10^9.

    # Consider A = N + 1 and M = N^2.
    # 1. A, M are positive: Since N >= 1, N+1 >= 2 and N^2 >= 1.
    # 2. A, M <= 10^18:
    #    If N = 10^9, A = 10^9 + 1, which is <= 10^18.
    #    If N = 10^9, M = (10^9)^2 = 10^18, which is <= 10^18.
    # 3. ord_M(A) = N?
    #    We need ord_{N^2}(N+1) = N.
    #    First, check gcd(A, M) = gcd(N+1, N^2).
    #    gcd(N+1, N^2) = gcd(N+1, N^2 mod (N+1)).
    #    Since N = -1 (mod N+1), N^2 = (-1)^2 = 1 (mod N+1).
    #    So gcd(N+1, N^2) = gcd(N+1, 1) = 1. A and M are coprime.
    #
    #    Now, consider powers of (N+1) mod N^2.
    #    (N+1)^k = sum_{i=0 to k} (k choose i) * N^i
    #             = (k choose 0)N^0 + (k choose 1)N^1 + (k choose 2)N^2 + ...
    #             = 1 + k*N + k(k-1)/2 * N^2 + ...
    #    Modulo N^2, this is (N+1)^k === 1 + k*N (mod N^2).
    #
    #    We want (N+1)^n === 1 (mod N^2).
    #    So, 1 + n*N === 1 (mod N^2).
    #    This means n*N === 0 (mod N^2).
    #    This implies N^2 must divide n*N.
    #
    #    If N=1: A=2, M=1. (N+1)^n - 1 is multiple of 1. This is true for n=1. ord_1(2)=1. Correct.
    #
    #    If N > 1: N^2 | n*N means N | n.
    #    So n must be a multiple of N. Let n = c*N for some positive integer c.
    #    The smallest such positive n would be when c=1, so n=N.
    #    We need to verify that (N+1)^N === 1 (mod N^2).
    #    (N+1)^N = 1 + N*N + N(N-1)/2 * N^2 + ...
    #             = 1 + N^2 + terms divisible by N^3 (or N^2 for N=2)
    #    So (N+1)^N === 1 (mod N^2).
    #    This shows that the order divides N.
    #
    #    Combined with n being a multiple of N, the smallest positive n is N itself.
    #    Thus, ord_{N^2}(N+1) = N for N > 1.
    #
    #    This construction (A=N+1, M=N^2) works for all N >= 1.

    A = N + 1
    M = N * N
    
    # Special case for N=1, to match sample's A value if one desires, but (2,1) is fine.
    # The logic A=N+1, M=N^2 gives (2,1) for N=1.
    # A=2, M=1. (2)^1 - 1 = 1 is a multiple of 1. Smallest n is 1.
    
    print(A, M)

T = int(input())
for _ in range(T):
    solve()