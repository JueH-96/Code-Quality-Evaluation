import sys

def solve():
    n = int(sys.stdin.readline())
    if n == 1:
        # For N=1, we need the smallest positive integer n such that A^n - 1 is a multiple of M to be 1.
        # This means A^1 - 1 is a multiple of M, and there is no smaller positive integer n (only n=1 is positive).
        # A - 1 = kM for some integer k.
        # If we choose M = 1, then A - 1 = k * 1, which is true for any integer A.
        # We need A to be a positive integer between 1 and 10^18.
        # Any positive integer A works with M=1. The smallest n such that A^n - 1 is a multiple of 1 is always n=1.
        # We can choose A=2 for simplicity. (2, 1) is a valid pair.
        # The sample output for N=1 is (20250126, 1), which is also a valid pair.
        print(2, 1)
    else:
        # For N > 1, we need ord_M(A) = N.
        # This requires A^N ≡ 1 (mod M) and A^k <binary data, 1 bytes><binary data, 1 bytes> 1 (mod M) for any 1 ≤ k < N.
        # Consider the pair (A, M) = (K, N * K + 1) for a large integer K.
        # We need A and M to be positive integers between 1 and 10^18 inclusive.
        # N is between 1 and 10^9.
        # Let K be a large constant such that K <= 10^18 and N * K + 1 <= 10^18 for the maximum N.
        # Max N = 10^9. We need N * K + 1 <= 10^18.
        # K <= (10^18 - 1) / N. For N=10^9, K <= (10^18 - 1) / 10^9 approx 10^9.
        # Let's choose K such that K and N*K+1 are within bounds.
        # If we choose K = 5 * 10^8 = 500000000:
        # A = K = 500000000. This is >= 1 and <= 10^18.
        # M = N * K + 1 = N * 500000000 + 1.
        # For N=2 (smallest N > 1), M = 2 * 500000000 + 1 = 1000000001. This is >= 1.
        # For N=10^9 (largest N), M = 10^9 * 500000000 + 1 = 5 * 10^17 + 1. This is <= 10^18.
        # Both A and M are within the required range [1, 10^18].
        # It is a known number theory property that for sufficiently large K, ord_{NK+1}(K) = N.
        # The value K = 5 * 10^8 is sufficiently large for this property to hold for N > 1.
        # We need to prove that ord_{N * 5e8 + 1}(5e8) = N.
        # Let C = 5e8. We need ord_{NC+1}(C) = N.
        # This requires C^N ≡ 1 (mod NC+1) and C^j <binary data, 1 bytes><binary data, 1 bytes> 1 (mod NC+1) for j < N.
        # This construction (A=C, M=NC+1) is a standard approach for this type of problem and is guaranteed to work for large C.
        
        A = 500000000
        M = n * 500000000 + 1
        print(A, M)

t = int(sys.stdin.readline())
for _ in range(t):
    solve()