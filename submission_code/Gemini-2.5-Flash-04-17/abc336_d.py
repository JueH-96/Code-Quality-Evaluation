import sys

def can_form_pyramid(k, N, A, R_prime, L_prime):
    """Checks if a pyramid of size k can be formed."""
    length = 2 * k - 1
    if length > N:
        return False

    # A pyramid of size k (1, 2, ..., k, ..., 2, 1)
    # can be formed from a contiguous subsegment A[i...i+length-1]
    # by decreasing values.
    # If we center the pyramid peak (value k) at index p in A,
    # the segment is A[p-(k-1) ... p+(k-1)].
    # The values required are:
    # A[p-j] >= k-j for j=0, ..., k-1 (left side: k, k-1, ..., 1)
    # A[p+j] >= k-j for j=0, ..., k-1 (right side: k, k-1, ..., 1)
    # The range of indices for the segment [p-(k-1), p+(k-1)] must be within [0, N-1].
    # p-(k-1) >= 0 => p >= k-1
    # p+(k-1) <= N-1 => p <= N-k
    # So the peak index p must be in the range [k-1, N-k].

    # R_prime[i] = max length m s.t. A[i-j] >= m-j for j=0..m-1 (sequence m, m-1, ..., 1 ending at A[i])
    # Calculated by DP: R_prime[i] = min(A[i], 1 + R_prime[i-1])
    # We need A[p-j] >= k-j for j=0..k-1, which is R_prime[p] >= k.

    # L_prime[i] = max length m s.t. A[i+j] >= m-j for j=0..m-1 (sequence m, m-1, ..., 1 starting at A[i])
    # Calculated by DP: L_prime[i] = min(A[i], 1 + L_prime[i+1])
    # We need A[p+j] >= k-j for j=0..k-1, which is L_prime[p] >= k.

    # So, for a fixed k, we need to check if there exists an index p
    # in the range [k-1, N-k] such that R_prime[p] >= k and L_prime[p] >= k.

    for p in range(k - 1, N - k + 1):
        if R_prime[p] >= k and L_prime[p] >= k:
            return True

    return False

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute R_prime array
    # R_prime[i] = max length m s.t. A[i], A[i-1], ..., A[i-m+1] can form m, m-1, ..., 1
    # A[i-j] >= m-j for j=0..m-1
    R_prime = [0] * N
    if N > 0:
        R_prime[0] = 1 if A[0] >= 1 else 0
        for i in range(1, N):
            if A[i] >= 1:
                R_prime[i] = min(A[i], R_prime[i-1] + 1)
            else:
                R_prime[i] = 0

    # Compute L_prime array
    # L_prime[i] = max length m s.t. A[i], A[i+1], ..., A[i+m-1] can form m, m-1, ..., 1
    # A[i+j] >= m-j for j=0..m-1
    L_prime = [0] * N
    if N > 0:
        L_prime[N-1] = 1 if A[N-1] >= 1 else 0
        for i in range(N - 2, -1, -1):
            if A[i] >= 1:
                L_prime[i] = min(A[i], L_prime[i+1] + 1)
            else:
                L_prime[i] = 0

    # Binary search for the maximum k
    # A pyramid of size k has length 2k-1. Max length is N.
    # 2k-1 <= N => 2k <= N+1 => k <= (N+1)/2.
    low = 1
    high = (N + 1) // 2
    max_k = 0

    while low <= high:
        mid = (low + high) // 2
        if can_form_pyramid(mid, N, A, R_prime, L_prime):
            max_k = mid
            low = mid + 1
        else:
            high = mid - 1

    print(max_k)

solve()